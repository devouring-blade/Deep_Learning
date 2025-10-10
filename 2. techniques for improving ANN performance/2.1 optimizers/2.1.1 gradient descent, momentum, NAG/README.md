# Gradient Descent 
gradient descent is the most basic optimizer and has the advantage of being easy to understand and implement. however, this has some 
disadvantage, such as being easily trapped in local minima or having an oscillating path to the target point.

<img width="1410" height="511" alt="{281FD03A-CB88-435E-BA32-5D9EE1265806}" src="https://github.com/user-attachments/assets/37b84aea-b252-4deb-9fca-92cfbd1f7f3c" />



# Momentum
the momentum optimizer adds momentum to gradient descent. it adds the current gradient to the previous momentum. this is the momentum vector. the vector remembers past gradients.

the parameter w is updated in the direction where the local gradient and previous momentum are combined.

this will accelerate the gradient descent towards the target point and dampen the oscillations.
beta is a hyper-parameter and is specified as a value between 0 and 1. the inertial of the distant past gradully weakens, and the inertial of the recent past takes on greater weight.

due to inertial, the momentum optimizer may pass through a shallow local minimum or may take a different route.

however, near the target point, we need to approach the target point slowly, but we can pass the target point due to acceleration. of course, in the next iteration we will return towards the target point.

<img width="1863" height="568" alt="{A73DD4D3-95D3-4E67-80B3-A7A194144EA2}" src="https://github.com/user-attachments/assets/413e358e-0489-4620-9b04-e2bfda10db8e" />

<img width="1235" height="659" alt="{BB0BF55E-0CCB-48A3-9F21-5F0237AA126B}" src="https://github.com/user-attachments/assets/81befdad-4517-4eff-8817-9349d39c80c1" />


# Nesterov Accelerated Gradient (NAG)
The momentum optimizer has a disadvantage near the target point that it may pass the target point due to inertia. NAG is an improvement on this.

When determining 𝑤𝑡+1, the momentum optimizer calculates the local gradient at the current point (A), while the NAG optimizer calculates the local gradient at the point where momentum was applied (B).


<img width="1846" height="708" alt="{6E1ABACE-9A99-4922-BAF5-260E0D4C047F}" src="https://github.com/user-attachments/assets/7702bbae-29b0-465d-839e-93434a91c423" />



NAG improves the disadvantage of the momentum optimizer, which may pass the target near the target point.

when w is far away from the target point, it can quickly approach the target point due to the momentum affect. and when w approaches the target point, it can stably reach the target point without deviating from the target point due to NAG effect

<img width="1733" height="547" alt="{BEF6CDCE-D787-476A-9D14-BA5C420FCBB9}" src="https://github.com/user-attachments/assets/a6efc408-c77c-4f7c-8fd1-ef0f2710ac1e" />



