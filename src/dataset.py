import torch


def build_sequence_dataset(trajectories):
    inputs = []
    targets = []

    for traj in trajectories:
        for i in range(1, len(traj)):
            inp = traj[:i]
            target = traj[i]

            inputs.append(inp)
            targets.append(target)

    return inputs, torch.tensor(targets)