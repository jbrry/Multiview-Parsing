# T=[en, fr, it, ja, ...]

# Initialize Φ, the initial parameter vector
# for i in meta_iters:
#   sample a task t from (T) (where the number of tasks sampled corresponds to (n-way, so 5-way would be 5 languages).
#     perform k steps of SGD on task t (starting with params Φ and resulting in params W)
#     meta gradient update: Φ←Φ+ϵ(W−Φ)
# end for
# Return Φ


self.data_loader = data_loader
self._validation_data_loader = validation_data_loader
self.optimizer = optimizer


def test_data_loader():

    batch_generator = iter(self.data_loader)

    batch_group_generator = common_util.lazy_groups_of(
        batch_generator, self._num_gradient_accumulation_steps
    )
