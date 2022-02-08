import torch

class CrossStitchBlock(torch.nn.Module):
    """
    Cross-stitch operation given input list of task outputs.
    """
    def __init__(self,
                 num_tasks: int = 2,
                 num_subspaces: int = 1):
        super(CrossStitchBlock, self).__init__()
        
        self.num_tasks = num_tasks
        self.num_subspaces = num_subspaces
        
        # initialize using random uniform distribution as suggested in Section 5.1
        self.cross_stitch_kernel = torch.FloatTensor(
                                                self.num_tasks,
                                                self.num_tasks).uniform_(
            0.0, 1.0
            ).to(device="cuda:0")
        # normalize, so that each row will be convex linear combination
        normalizer = torch.sum(self.cross_stitch_kernel, 0, keepdim=True)
        self.cross_stitch_kernel = self.cross_stitch_kernel / normalizer
    

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # concatenate nth element of each input task
        x = torch.cat([torch.unsqueeze(i, -1) for i in inputs], -1)

	# multiply every element of the input with the cross-stitch kernel
        stitched_output = torch.matmul(x, self.cross_stitch_kernel)

        n, batch, seq, _ = x.size()
        # index need to be of the same dimension size as source tensor
        index = torch.zeros(n, batch, seq, 1, dtype=torch.int64).to(device="cuda:0")

        # split result into tensors corresponding to specific tasks and return
        # to task-specific lists
        outputs = [
            torch.flatten(torch.gather(e, dim=-1, index=index), start_dim=2)
            for e in torch.split(stitched_output, 1, -1)
        ]

        assert len(outputs) == self.num_tasks
        return outputs
