

An = f(Wn*An-1 + b), Un = Wn*An-1 + b

input [dJ / d An]:

calc weight gradients and update:  
(1) dJ / dWn =  [dJ / dAn] *   [d f(Un) / dUn]  *   [d Un / d Wn]

calc bias gradients and update
(2) dJ / db


calc input to prev layer
(3) dJ / dAn-1 = [dJ / dAn]  *  [d f(Un) / d Un]   * [d Un / dAn-1] 

verify: 
gradW = ( input . An-1 transpose) *  f'(Un) / m 
gradB = sum(input * f'(U(n)) (i)) / m ; input(i) = ith column of input
gradA = (W transpose . input) *  f'(Un) 

