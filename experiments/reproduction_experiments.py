import uea
import sepsis
import speech_commands

# group = 1 corresponds to 30% mising rate
uea.run_all(group=1, device='cuda:3', dataset_name='CharacterTrajectories')
# group = 2 corresponds to 50% mising rate
uea.run_all(group=2, device='cuda:3', dataset_name='CharacterTrajectories')
# group = 3 corresponds to 70% mising rate
uea.run_all(group=3, device='cuda:3', dataset_name='CharacterTrajectories')

sepsis.run_all(intensity=True, device='cuda:3')
sepsis.run_all(intensity=False, device='cuda:3')

speech_commands.run_all(device='cuda:3')