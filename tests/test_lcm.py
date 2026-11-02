"""CPU-only synthetic tests; set COMFYUI_PATH to an installed ComfyUI source tree."""
import importlib.util
import os
import sys
import threading
import types
import unittest
from unittest.mock import patch

sys.argv.append('--cpu')
sys.path.insert(0, os.environ['COMFYUI_PATH'])
import comfy.options
comfy.options.enable_args_parsing()
import torch
import comfy.k_diffusion.sampling as sampling
import comfy.model_sampling
import comfy.nested_tensor
import comfy.samplers
import comfy.utils

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location('selflift_test', os.path.join(root, '__init__.py'), submodule_search_locations=[root])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
from selflift_test import nodes, diagnostics
sys.argv.remove('--cpu')

class Identity:
    spacial_downscale_ratio = 1
    def process_in(self, x): return x
    def process_out(self, x): return x
    def process_latent_out(self, x): return x

class Model:
    load_device = torch.device('cpu')
    model_options = {}
    model = Identity()
    def __init__(self):
        self.sampling = comfy.model_sampling.CONST()
        self.format = Identity()
    def get_model_object(self, name):
        return {'model_sampling': self.sampling, 'latent_format': self.format}[name]

class LCMTests(unittest.TestCase):
    def test_progressive_nested_and_no_lift_matches_native(self):
        for scale in (1.0, 0.5):
            with self.subTest(scale=scale):
                model = Model()
                video = torch.zeros(1, 2, 2, 8, 8)
                audio = torch.zeros(1, 2, 5)
                latent = {'samples': comfy.nested_tensor.NestedTensor([video, audio])}
                sigmas = torch.tensor([.95, .8, .65, .5, .35, .2, 0.])
                calls, shapes, states = [], [], []
                def denoise(x, sigma, **kwargs):
                    calls.append(float(sigma[0])); shapes.append(tuple(x.shape)); states.append(x.clone())
                    return x * .7 + .05
                denoise.inner_model = types.SimpleNamespace(model_patcher=model)
                def fake_sample(m, noise, positive, negative, cfg, device, sampler, schedule, options, latent_image, callback, **kwargs):
                    streams = list(latent_image.unbind())
                    packed, stream_shapes = comfy.utils.pack_latents(streams)
                    noise, _ = comfy.utils.pack_latents(noise.unbind())
                    x = m.sampling.noise_scaling(schedule[0], noise, packed)
                    def cb(d):
                        callback(d['i'], comfy.nested_tensor.NestedTensor(comfy.utils.unpack_latents(d['denoised'], stream_shapes)),
                                 comfy.nested_tensor.NestedTensor(comfy.utils.unpack_latents(d['x'], stream_shapes)), len(schedule)-1)
                    out = sampler.sampler_function(denoise, x, schedule, extra_args={'seed': 42}, callback=cb, disable=True, **sampler.extra_options)
                    out = m.sampling.inverse_noise_scaling(schedule[-1], out)
                    return comfy.nested_tensor.NestedTensor(comfy.utils.unpack_latents(out, stream_shapes))
                lifter_calls = []
                def lift(x, hw):
                    lifter_calls.append(tuple(x.shape))
                    return torch.nn.functional.interpolate(x, size=(x.shape[2], *hw), mode='nearest')
                with patch.object(nodes.comfy.samplers, 'sample', fake_sample), patch.object(nodes.comfy.sample, 'fix_empty_latent_channels', lambda model, samples, *args: samples), patch.object(nodes.latent_preview, 'prepare_callback', lambda *args: lambda *args: None):
                    result = nodes.progressive_sample(model, [], [], None, latent, comfy.samplers.KSAMPLER(sampling.sample_lcm), sigmas, 42, 1, 3, scale, 0, 0, 0, 'nearest', latent_lifter=lift, allow_lcm=True)
                self.assertEqual(len(calls), 6)
                self.assertEqual(lifter_calls != [], scale != 1)
                self.assertEqual(len(shapes), 6)
                self.assertEqual(tuple(list(result['samples'].unbind())[0].shape), tuple(video.shape))
                if scale == .5:
                    # Independently reconstruct the boundary from native LCM's
                    # first three predictions and its single persistent RNG.
                    low_video = torch.zeros(1, 2, 2, 4, 4)
                    low_latent = comfy.nested_tensor.NestedTensor([low_video, audio])
                    low_packed, low_shapes = comfy.utils.pack_latents(low_latent.unbind())
                    low_noise, _ = comfy.utils.pack_latents(comfy.sample.prepare_noise(low_latent, 42).unbind())
                    state = model.sampling.noise_scaling(sigmas[0], low_noise, low_packed)
                    generator = torch.Generator().manual_seed(43)
                    for i in range(3):
                        x0 = state * .7 + .05
                        if i < 2:
                            noise = torch.randn(state.shape, generator=generator)
                            state = model.sampling.noise_scaling(sigmas[i+1], noise, x0)
                    low_x0 = comfy.utils.unpack_latents(x0, low_shapes)
                    lifted = torch.nn.functional.interpolate(low_x0[0], size=(2, 8, 8), mode='nearest')
                    high_x0, _ = comfy.utils.pack_latents([lifted, low_x0[1]])
                    noise = torch.randn(high_x0.shape, generator=generator)
                    expected = model.sampling.noise_scaling(sigmas[3], noise, high_x0)
                    self.assertTrue(torch.allclose(states[3], expected, atol=1e-6))
                if scale == 1:
                    calls.clear()
                    packed, _ = comfy.utils.pack_latents([video, audio])
                    initial_noise = comfy.sample.prepare_noise(latent['samples'], 42)
                    initial_noise, _ = comfy.utils.pack_latents(initial_noise.unbind())
                    initial = model.sampling.noise_scaling(sigmas[0], initial_noise, packed)
                    native = sampling.sample_lcm(denoise, initial, sigmas, extra_args={'seed':42}, disable=True)
                    actual, _ = comfy.utils.pack_latents(result['samples'].unbind())
                    self.assertTrue(torch.allclose(actual, native, atol=1e-6), (actual-native).abs().max())

    def test_lcm_mask_rejected(self):
        model = Model()
        latent = {'samples': torch.zeros(1, 2, 2, 8, 8), 'noise_mask': torch.ones(1, 8, 8)}
        with self.assertRaisesRegex(ValueError, 'noise_mask'):
            nodes.progressive_sample(model, [], [], None, latent, comfy.samplers.KSAMPLER(sampling.sample_lcm), torch.tensor([.9, .5, 0.]), 1, 1, 1, .5, 0, 0, 0, 'nearest', allow_lcm=True)

class MMH3IntegrationTests(unittest.TestCase):
    def test_disabled_returns_original_without_resolving_mmh3(self):
        original = {'samples': object()}
        with patch.object(nodes, 'progressive_sample', return_value=original):
            out = nodes.SelfLiftH3Sampler().sample(
                object(), [], [], None, original, object(), object(), 42, 3.0,
                1, .5, .5, .5, 1.0, 'none')
        self.assertIs(out[0], original)
        self.assertIs(out[1], original)

    def test_enabled_forwards_patcher_sampling_and_options(self):
        original, final = {'samples': object()}, {'samples': object()}
        base, hires, pos, neg = object(), object(), object(), object()
        noise, sampler, sigmas = object(), object(), object()
        upscale, temporal, spatial = object(), object(), object()
        calls = []
        class MMH3:
            @classmethod
            def execute(cls, offload_diffusion=True, **kwargs):
                calls.append(dict(kwargs, offload_diffusion=offload_diffusion))
                return (final, {}, {})
        import nodes as comfy_nodes
        with patch.dict(comfy_nodes.NODE_CLASS_MAPPINGS, {'MMH3UltimateUpscale': MMH3}), \
             patch.object(nodes, 'progressive_sample', return_value=original):
            out = nodes.SelfLiftH3Sampler().sample(
                base, pos, neg, None, original, object(), object(), 42, 3.0,
                1, .5, .5, .5, 1.0, 'none', model_hires=hires, mmh3_enabled=True,
                mmh3_noise=noise, mmh3_sampler=sampler, mmh3_sigmas=sigmas,
                latent_upscale_param=upscale, temporal_split_param=temporal, spatial_split_param=spatial)
        self.assertEqual(out, (original, final))
        self.assertEqual(calls, [dict(latent=original, conditioning=pos, model=hires,
            noise=noise, sampler=sampler, sigmas=sigmas, negative=neg, cfg=3.0,
            latent_upscale_param=upscale, temporal_split_param=temporal,
            spatial_split_param=spatial, offload_diffusion=False)])

class DiagnosticTests(unittest.TestCase):
    def model(self):
        return types.SimpleNamespace(wrappers={}, model_options={}, model=object(),
                                     patches={}, patches_uuid='same-weights', load_device=torch.device('cpu'))

    def test_native_wrappers_preserve_identity_outputs_and_callback(self):
        model = self.model()
        original_wrappers, original_options, base = model.wrappers, model.model_options, model.model
        callback_calls = []
        owner = types.SimpleNamespace(model_patcher=model)
        extension = nodes.comfy.patcher_extension
        def invoke(kind, fn, *args):
            wrappers = [wrapper for group in model.wrappers[kind].values() for wrapper in group]
            return extension.WrapperExecutor.new_class_executor(fn, owner, wrappers).execute(*args)
        def predict(x):
            return x + 2
        def prepare(patcher):
            self.assertIs(patcher, model)
            return patcher.model
        def sample(guider, sigmas, options, callback, noise):
            result = invoke('predict_noise', predict, noise)
            self.assertEqual(callback(0, result, noise, 1), 'callback-result')
            return result
        def outer():
            self.assertIs(invoke('prepare_sampling', prepare, model), base)
            return invoke('sampler_sample', sample, owner, object(), {}, callback, 3)
        def callback(*args):
            callback_calls.append(args)
            return 'callback-result'
        with patch.dict(os.environ, {'SELFLIFT_DIAGNOSTICS': '1'}), patch.object(diagnostics.logging, 'info') as log:
            with diagnostics.diagnostic_scope('test', model):
                self.assertEqual(invoke('outer_sample', outer), 5)
            self.assertTrue(any('compile_s=unknown' in str(call) for call in log.call_args_list))
        self.assertEqual(callback_calls, [(0, 5, 3, 1)])
        self.assertIs(model.wrappers, original_wrappers)
        self.assertIs(model.model_options, original_options)
        self.assertIs(model.model, base)
        self.assertEqual(model.patches_uuid, 'same-weights')
        self.assertIsNone(diagnostics._active_stage.get())
        self.assertIsNone(diagnostics._active_sample.get())

    def test_error_restores_scope_and_does_not_sync_cuda(self):
        model = self.model()
        original = model.wrappers
        with patch.dict(os.environ, {'SELFLIFT_DIAGNOSTICS': '1'}), \
             patch.object(diagnostics.torch.cuda, 'synchronize') as synchronize, \
             patch.object(diagnostics.psutil, 'Process', side_effect=OSError('probe unavailable')):
            with self.assertRaisesRegex(RuntimeError, 'sampling failed'):
                with diagnostics.diagnostic_scope('test', model):
                    def failed_sample():
                        raise RuntimeError('sampling failed')
                    executor = nodes.comfy.patcher_extension.WrapperExecutor.new_class_executor(
                        failed_sample, types.SimpleNamespace(model_patcher=model), [diagnostics._outer_sample])
                    executor.execute()
        self.assertIs(model.wrappers, original)
        self.assertIsNone(diagnostics._active_stage.get())
        self.assertIsNone(diagnostics._active_sample.get())
        self.assertFalse(diagnostics._scopes)
        synchronize.assert_not_called()

    def test_overlapping_threads_restore_only_after_last_scope(self):
        model = self.model()
        original = model.wrappers
        entered, release = threading.Event(), threading.Event()
        failures = []
        def worker():
            try:
                with diagnostics.diagnostic_scope('worker', model):
                    entered.set()
                    if not release.wait(5):
                        raise AssertionError('scope release timed out')
            except BaseException as error:
                failures.append(error)
        with patch.dict(os.environ, {'SELFLIFT_DIAGNOSTICS': '1'}):
            thread = threading.Thread(target=worker)
            thread.start()
            try:
                self.assertTrue(entered.wait(5))
                installed = model.wrappers
                # A reader in an unrelated context sees wrappers that delegate unchanged.
                self.assertEqual(diagnostics._predict_noise(lambda value: value + 1, 3), 4)
                with diagnostics.diagnostic_scope('main', model):
                    self.assertIs(model.wrappers, installed)
                self.assertIs(model.wrappers, installed)
            finally:
                release.set()
                thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertFalse(failures)
        self.assertIs(model.wrappers, original)
        self.assertFalse(diagnostics._scopes)

    def test_disabled_does_not_install_wrappers(self):
        model = self.model()
        original = model.wrappers
        with patch.dict(os.environ, {'SELFLIFT_DIAGNOSTICS': '0'}), patch.object(diagnostics.logging, 'info') as log:
            with diagnostics.diagnostic_scope('disabled', model) as stage:
                stage.event('unused')
                self.assertIs(model.wrappers, original)
            log.assert_not_called()

if __name__ == '__main__': unittest.main()
