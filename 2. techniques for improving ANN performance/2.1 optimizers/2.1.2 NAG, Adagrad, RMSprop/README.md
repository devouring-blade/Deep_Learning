# Nesterov Accelerated Gradient (NAG)
The momentum optimizer has a disadvantage near the target point that it may pass the target point due to inertia. NAG is an improvement on this.

When determining 𝑤𝑡+1, the momentum optimizer calculates the local gradient at the current point (A), while the NAG optimizer calculates the local gradient at the point where momentum was applied (B).


<img width="1846" height="708" alt="{6E1ABACE-9A99-4922-BAF5-260E0D4C047F}" src="https://github.com/user-attachments/assets/7702bbae-29b0-465d-839e-93434a91c423" />


<img width="1716" height="565" alt="{326234FE-7A45-4057-9680-88FB4EE034F4}" src="https://github.com/user-attachments/assets/d1e03cce-d97e-446f-8a5e-c20ac01e65f7" />
