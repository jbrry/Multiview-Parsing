import torch


def cross_stitch(inputs):
    """
    Cross stitch layer, takes as input a list of task specific tensors.

    inputs: List[Tensor]:
        List of Tensor objects, where each Tensor object is the output of a certain task.
        The length of the list is equal to the number of tasks,
        and contain the outputs of a particular layer from each task.

    Returns:
        outputs: List[Tensor]:
        List of length of the number of tasks containing the linear combination of
        predictions for each task. E.g. for 2 tasks, a list containing 2 lists.

    """

    stitch_shape = len(inputs)
    print("\n")
    print(f"stitch shape {stitch_shape}")

    # initialize using random uniform distribution as suggested in section 5.1
    cross_stitch_kernel = torch.FloatTensor(stitch_shape, stitch_shape).uniform_(
        0.0, 1.0
    )
    # normalize, so that each row will be convex linear combination,
    # here we follow recommendation in paper (see section 5.1 )
    normalizer = torch.sum(cross_stitch_kernel, 0, keepdim=True)
    cross_stitch_kernel = cross_stitch_kernel / normalizer

    print(f"the cross stitch kernel after normalization: \n {cross_stitch_kernel}")
    print()

    # concatenate each element of the input for each task
    # this will contain the nth element of each task
    x = torch.cat([torch.unsqueeze(i, -1) for i in inputs], -1)
    print("x", x, x.size())
    print()

    # multiply every element of the input with the cross-stitch kernel
    stitched_output = torch.matmul(x, cross_stitch_kernel)
    print(f"stitched_output:\n{stitched_output} {stitched_output.size()}")

    n, batch, seq, d = x.size()
    # index need to be of the same dimension size as source tensor
    index = torch.zeros(n, batch, seq, 1, dtype=torch.int64)

    # split result into tensors corresponding to specific tasks and return
    # to task-specific lists
    outputs = [
        torch.flatten(torch.gather(e, dim=-1, index=index), start_dim=2)
        for e in torch.split(stitched_output, 1, -1)
    ]

    assert len(outputs) == stitch_shape

    return outputs


inputs = [torch.randn(1, 2, 3) for _ in range(2)]

# task_1 = torch.randn(8, 10, 50)
# task_2 = torch.randn(8, 10, 50)
# task_3 = torch.randn(8, 10, 50)
# task_4 = torch.randn(8, 10, 50)
# inputs = [task_1, task_2, task_3, task_4]

print(f"Running test with inputs: {inputs}")
# get linear combination between each task and the cross-stitch units
output = cross_stitch(inputs)
# unpack cross-stitched output to be passed to subsequent layer
task_1, task_2, task_3, task_4 = output
