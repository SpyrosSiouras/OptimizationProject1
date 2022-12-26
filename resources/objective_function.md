# Αντικειμενική συνάρτηση

Ας είναι
$$p_t(a):= \displaystyle\sum_{i=0}^n t^ia_i$$
$$\text{και}$$
$$f(a):=\displaystyle\frac{1}{N} \sum_{t=1}^N \left( p_t(a)-y_t\right)^2$$
όπου $n$, $N\in\N$, $t\in\N\cap[1,N]$, $a\in\R^n$ και $\bigl(y_t\bigr)_{t=1,\dots,Ν}\in\R^N$.

Υπάρχουν σταθερές $M\in\R^{n\times n}$, $v\in\R^n$, $y\in\R^Ν$ ανεξάρτητες του $a$ και τέτοιες ώστε: $$f(a) = \frac{1}{2}~ a^T\cdot M\cdot a + ~v^T\cdot a +  y^2$$

Πράγματι, αρκεί να θέσουμε:
$$
  \begin{aligned}
    M&:=\frac{2}{N}\left(\sum_{t=1}^Nt^{i+j}\right)_{i,j=1,\dots,n}\in\R^{n\times n} \\
    v&:=-\frac{2}{N}\left(\sum_{t=1}^Ny_t t^i \right)_{i=1,\dots,n}\in\R^{n} \\
    y&:=\frac{1}{\sqrt{N}} \bigl(y_t\bigr)_{t=1,\dots,N}\in\R^N
  \end{aligned}
$$

Τότε ισχύει:
$$\nabla f(a) = M \cdot a + v$$
$$D^2 f(a) = M$$

Για να τεστάρουμε την υλοποίηση της συνάρτησης έχουμε τις ταυτότητες:

$$f(a) = 0, \quad\text{όταν } y_t=p_t(a)~~ \forall t=1,\dots,N$$
$$f(a) = \frac{1}{2} ~a^T\cdot M \cdot a, \quad\text{όταν } y_t=0~~ \forall t=1,\dots,N$$
$$f(0) = y^2 = \frac{1}{N} \sum_{t=1}^N y_t^2$$
$$f(a) = f(0) + \nabla^Tf(0)\cdot a + \frac{1}{2}~a^T\cdot D^2f(0)\cdot a $$

Τέλος, λύνοντας το σύστημα

$$\nabla f(a_0) = 0 \iff a_0 = -M^{-1}\cdot v$$

βρίσκουμε υποψήφια σημείο για ύπαρξη ακροτάτου

<!-- + $$
    \begin{aligned}
        p(t,a) :=& ~ a_0 + a_1 t + a_2 t^2 + a_3 t^3 + a_4 t^4 \\
                =& ~ v^T_t \cdot~ a~ = ~a^T\cdot ~v_t \\
    \end{aligned}
  $$
  όπου $a := (a_0,a_1,a_2,a_3,a_4)^T$ και $v_t := (1,t,t^2,t^3,t^4)^T$.

+ $$
    \begin{aligned}
        p^2(t,a) =& ~ a^T \cdot M_t \cdot a\\
                 =& ~\sum_{i,j=0}^4 ~ m^t_{ij} a_i a_j\\
    \end{aligned}
  $$
  όπου $m^t_{ij} := t^{i+j}$ και $M_t =\Bigl( ~ m^t_{ij} ~ \Bigr)_{i,j=0}^4$

+ $$
    \begin{aligned}
        f(x) :=& \frac{1}{25} \sum_{t=1}^{25} ~ ( ~ p(t,x) ~ - ~ y_t ~ )^2 \\
              =& \frac{1}{25} \sum_{t=1}^{25} ~ p^2(t,x) ~ - 2y_tp(t,x) ~ + ~ y_t^2 \\
              =& ~ x^T \cdot M \cdot x ~ + ~2~ v^T \cdot x ~ + ~{y^2}
    \end{aligned}
  $$
  όπου $M := \displaystyle \frac{1}{25}\sum_{t=1}^{25} M_t$,
  $~v := \displaystyle -\frac{1}{25}\sum_{t=1}^{25} ~ y_t~v_t$
  και $y^2 := \displaystyle \frac{1}{25}\sum_{t=1}^{25} y_t^2$.

Συνεπώς έχουμε:

 $$ f(x) = ~ x^T \cdot M \cdot x ~+~2~~ v^T \cdot x + y^2 $$
 $$ \nabla f(x) = 2~M \cdot x + 2~v  $$
 $$ D^2f(x) = 2~M $$

Μάλιστα, ισχύει:

$$f(x+h) = f(x) +  \nabla^Tf(x)\cdot h + \frac{1}{2}~h^T\cdot D^2f(x) \cdot h $$ -->
