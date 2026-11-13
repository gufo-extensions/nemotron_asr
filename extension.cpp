#include <gufo/ext/asr.hpp>
#include <gufo/ext/registry.hpp>

#include "probe.hpp"

// Without HIP the provider's library isn't built; registering its probe
// still lets Gufo name the model it can't run here.
void nemotron_asr_register(gufo::ext::Registry& registry) {
#if !defined(ENGINE_ENABLE_HIP)
  registry.AddAsrProvider({
      .id = "nemotron-asr",
      .probe = &gufo::models::nemotron_asr::ProbeNemotronAsr,
      .resolve = &gufo::models::nemotron_asr::ResolveNemotronAsrArtifacts,
      .create = nullptr,
  });
#else
  (void)registry;
#endif
}
