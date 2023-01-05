# Optimization Project


> University of Ioannina -- Department of Computer Science & Engineering
> Graduate Studies
> (D3) Optimization


<center> <b> Finding optimal polynomial </b> </center>
<br>


Papazis Steve
Siouras Spyros 


### 1 Introduction
The purpose of this project is to fit a fourth degree polynomial to a given time series using techniques from numerical optimisation.

The fourth degree polyonimal: 
 $$\displaystyle P_x(t) = \sum_{i=0}^4x_it^i$$
 for every $x\in\mathbb{R}^5$. 

The objective function: 
$$f(x):=\frac{1}{m}\sum_{t=1}^m (y_t-P_x(t))^2$$



### 2 Algorithms
  We implemented the line search methods Steepest Descent, Newton and BFGS with strong wolf conditions, as well as a dogleg version of the BFGS method. As a standard for implementations we had the material of the slides, but also the book Numerical Optimization by J.NoCedal and S.J.Wright. We used Python 3.9.5 for the implementation of the algorithms. 

  #### 2.1 Parameters: 

  **Wolf conditions**:
  
  $c_1=10^{-4},$ $c_2=0.9$, $a_{max} = 1,$ and $a_0 = 0$ for Steepest Descent.

      
  $c_1=10^{-4},$ $c_2=0.9$, $a_{max} = 1,$ and $a_0 = 1$ for Newton and BFGS.

**Dogleg**:

Trust region maximum radius is equal to 1. Initial value is equal to the maximun radius and $n=0.001$.


**General**:
For reproducubility purposes the random generator of numpy was used with the seed equal to 0. Last, but not least the termination conditions are the following: 

-  $x^\ast$ is found, for which $\| \nabla f(x^\ast) \| \le{} 10^{-8}$ 

- function calls surpassed 3000. 




### 3 Instructions

To install needed packages:
**pip install requirements.txt**

To run the code:
**python main.py**