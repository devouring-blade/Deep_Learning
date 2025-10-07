# gradient descent optimizer  
gradient descent is the most basic optimizer and has the advantage of being easy to understand and implement. however, this has some 
disadvantage, such as being easily trapped in local minima or having an oscillating path to the target point.

<img width="1876" height="676" alt="{E76904C2-EBAA-4379-A34F-3016117188CF}" src="https://github.com/user-attachments/assets/da50a87f-fbde-4102-8171-3b1f64c40535" />

# momentum optimizer
the momentum optimizer adds momentum to gradient descent. it adds the current gradient to the previous momentum. this is the momentum vector. the vector remembers past gradients.

the parameter w is updated in the direction where the local gradient and previous momentum are combined.

this will accelerate the gradient descent towards the target point and dampen the oscillations.
beta is a hyper-parameter and is specified as a value between 0 and 1. the inertial of the distant past gradully weakens, and the inertial of the recent past takes on greater weight.

due to inertial, the momentum optimizer may pass through a shallow local minimum or may take a different route.

however, near the target point, we need to approach the target point slowly, but we can pass the target point due to acceleration. of course, in the next iteration we will return towards the target point.

<img width="1863" height="568" alt="{A73DD4D3-95D3-4E67-80B3-A7A194144EA2}" src="https://github.com/user-attachments/assets/413e358e-0489-4620-9b04-e2bfda10db8e" />

# zig-zag oscillation
<img width="1684" height="875" alt="{DFCD6859-AE87-4C0F-AB0E-948D6E77B89F}" src="https://github.com/user-attachments/assets/574e3c0b-46f7-4317-ba72-345060985589" />




