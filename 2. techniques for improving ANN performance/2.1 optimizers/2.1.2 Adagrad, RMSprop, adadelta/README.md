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
Adadelta is an optimization algorithm proposed by Matthew D. Zeiler in a 2012 paper.

<img width="1884" height="927" alt="{166F126F-181F-4E3A-BF1C-92CD575EF198}" src="https://github.com/user-attachments/assets/695507dd-46d4-45fd-a7f6-dc56aff55995" />

In the ADAGRAD method the denominator accumulates the squared gradients from each iteration starting at the beginning of training. Since each term is positive, this accumulated sum continues to grow throughout training, effectively shrinking the learning rate on each dimension. After many iterations, this learning rate will become infinitesimally small.

### idea 1: accumulate over window
Instead of accumulating the sum of squared gradients over all time, we restricted the window of past gradients that are accumulated to be some fixed size T (instead of size t where t is the current iteration as in ADAGRAD). With this windowed accumulation the denominator of ADAGRAD cannot accumulate to infinity and instead becomes a local estimate using recent gradients. This ensures that learning continues to make progress even after many iterations of updates have been done. Since storing T previous squared gradients is inefficient, our methods implements this accumulation as an exponentially decaying average of the squared gradients.

<img width="1779" height="337" alt="{76CC00DD-3F63-44EF-A05D-03FBCB4E7E7E}" src="https://github.com/user-attachments/assets/7376befb-eb35-433c-9af1-9fa0c013413b" />

In fact, this is essentially the same as the RMSProp optimizer mentioned above. It was not published in a paper, but was proposed by Geoffrey Hinton in his Coursera lecture. Perhaps the author of the paper was thinking the same at that time.

### idea 2: correct units with Hessian Approximation 
When considering the parameter updates, Δw, being applied to w, the units should match. That is, if the parameter had some hypothetical units, the changes to the parameter should be changes in those units as well. When considering SGD, Momentum, or ADAGRAD, we can see that this is not the case. The units in SGD and Momentum relate to the gradient, not the parameter.

<img width="891" height="86" alt="{420BFE78-7233-4518-9F03-5319D646D320}" src="https://github.com/user-attachments/assets/6b17c79d-98d0-4303-8cbf-aa495f46b4bd" />

Assuming the cost function, L, is unitless. ADAGRAD also does not have correct units since the update involves ratios of gradient quantities, hence the update is unitless. In contrast, second order methods such as Newton’s method that use Hessian information or an approximation to the Hessian do have the correct units for the parameter updates.

<img width="871" height="173" alt="{41722EB9-EF07-4CE8-8A0F-63BD55C18646}" src="https://github.com/user-attachments/assets/a06f7ffa-0dd4-4678-b30c-e04ab3207d63" />

Noticing this mismatch of units we considered terms to add to Eqn. 10 in order for the units of the update to match the units of the parameters. Since second order methods are correct, we rearrange Newton’s method (assuming a diagonal Hessian) for the inverse of the second derivative to determine the quantities involved.

<img width="999" height="165" alt="{4FCDA6ED-665E-4147-8F6C-02445F7B5106}" src="https://github.com/user-attachments/assets/2021c809-02d9-4ced-a077-68101a5298d6" />

Since the RMS of the previous gradients is already represented in the denominator in equation 10, we considered a measure of the Δw quantity in the numerator. Δwt for the current time step is not known, so we assume the curvature is locally smooth and approximate Δwt by computing the exponentially decaying RMS over a window of size T of previous Δw to give the ADADELTA method.

<img width="1010" height="417" alt="{45610CB3-470F-4DE9-ADD2-5082E482C892}" src="https://github.com/user-attachments/assets/8a963118-2210-4c77-853a-6db7dc710a2d" />






