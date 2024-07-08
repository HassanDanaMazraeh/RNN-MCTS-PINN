 % Expression= ((w(0)*(x*x))+w(1)*w(2))
 % 
 % Parameters= tensor([-0.16138416528701782227, -0.98353004455566406250, -0.98389834165573120117])
 % 
 % Loss= 0.0002932326460722834
 % 
 % Main Expression in PyTorch= ((self.w[0]*(x*x))+self.w[1]*self.w[2])
 % 
 % Branch= [1, 1, 3, 1, 3, 1, 3, 0, 3, 3, 0, 0, 0, 3, 3, 3, 3, 3]
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 % After fine-tuning
clear
clc
syms x

y_exact=1-x^2/6;

w=[-0.16666665673255920410, -0.99982750415802001953, -1.00017178058624267578];
y_symbolic=((w(1)*(x*x))+w(2)*w(3));

d=[0:0.01:5];
f1=vpa(subs(y_symbolic,x,d));
f2=vpa(subs(y_exact,x,d));
p=plot(d,f1,d,f2);, grid on,legend('Symbolic solution','Exact solution');
p(1).LineWidth = 2;
p(2).LineWidth = 2;
set(gca, 'FontSize', 16, 'FontWeight', 'bold')

Roots_symbolic=vpasolve(y_symbolic,2);
Roots_exact=vpasolve(y_exact,2);
Root_exact=Roots_exact(2);

for i=1:length(w)
    fprintf('w(%d) & %.20f line \n',i,w(i));
end

y_symbolic_simplified=vpa(simplify(y_symbolic))
Root_symbolic=Roots_symbolic(2)
MAE=vpa(mean(abs(f1-f2)))