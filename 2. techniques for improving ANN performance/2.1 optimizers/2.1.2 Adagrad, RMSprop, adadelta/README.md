# Adaptive Gradient (Adagrad)
Gradient descent and momentum optimizers use a constant learning rate, and apply the same learning rate for all w. It is also a good idea to gradually reduce the learning rate while iterating. At the beginning of the iteration, w is likely to be far away from the target point, so w needs to be updated a lot by applying a large learning rate. And as w approaches the target point, w needs to be updated little by little by applying a smaller learning rate.

Adagrad not only does this, but also applies a different learning rate to each w depending on the magnitude of its gradient.

ϵ is to prevent the denominator from being zero. [G₀ = 0, ϵ = small value (ex: 10⁻⁶)]

As iterations progress, Gₜ increases exponentially, which has the disadvantage of decreasing the learning rate too quickly. 

<img width="1816" height="641" alt="{75F1005B-E949-413B-8088-36E2CE9584EA}" src="https://github.com/user-attachments/assets/64261034-4880-42d8-bdf2-439f8eea372d" />


# Root Mean Square Propagation (RMSprop)
RMSprop improves the disadvantage of Adagrad that the learning rate decays too quickly. When calculating Gt, we take the exponentially weighted average to the previous G and the most recent squared gradient, allowing the learning rate to decay smoothly. The larger ρ, the more past G is reflected, so the learning rate decreases more smoothly.

<img width="1858" height="744" alt="{40854B28-C4AD-4AD5-8B1F-BF2216837F2A}" src="https://github.com/user-attachments/assets/ab7ced7b-d716-4411-9b18-19324dce0971" />


# Adadelta
