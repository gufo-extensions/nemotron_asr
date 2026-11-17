#include <gufo/ext/asr.hpp>
#include <gufo/ext/generated/nemotron_asr_constants.hpp>
#include <gufo/ext/registry.hpp>

#include "probe.hpp"
#include "runner.hpp"

void nemotron_asr_provider_register(gufo::ext::Registry& registry) {
  namespace asr = gufo::models::nemotron_asr;
  registry.AddProvider(
      gufo::ext::generated::extension_nemotron_asr::kProvider_nemotron_asr);
  registry.AddAsrProvider({
      .id = "nemotron-asr",
      .probe = &asr::ProbeNemotronAsr,
      .resolve = &asr::ResolveNemotronAsrArtifacts,
      .create = &asr::CreateNemotronAsrRunner,
      .create_with_options = &asr::CreateNemotronAsrRunnerWithOptions,
  });
}
