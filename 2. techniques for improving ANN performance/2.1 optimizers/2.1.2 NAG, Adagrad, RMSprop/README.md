# Nesterov Accelerated Gradient (NAG)
The momentum optimizer has a disadvantage near the target point that it may pass the target point due to inertia. NAG is an improvement on this.

When determining 𝑤𝑡+1, the momentum optimizer calculates the local gradient at the current point (A), while the NAG optimizer calculates the local gradient at the point where momentum was applied (B).


<img width="1846" height="708" alt="{6E1ABACE-9A99-4922-BAF5-260E0D4C047F}" src="https://github.com/user-attachments/assets/7702bbae-29b0-465d-839e-93434a91c423" />

NAG improves the disadvantage of the momentum optimizer, which may pass the target near the target point.

when w is far away from the target point, it can quickly approach the target point due to the momentum affect. and when w approaches the target point, it can stably reach the target point without deviating from the target point due to NAG effect

<img width="1733" height="547" alt="{BEF6CDCE-D787-476A-9D14-BA5C420FCBB9}" src="https://github.com/user-attachments/assets/a6efc408-c77c-4f7c-8fd1-ef0f2710ac1e" />

