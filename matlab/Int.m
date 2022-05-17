function I=Intfunc(t,diag,F)
    x=log(t);
    dx=x[2]-x[1];
    I=1/sqrt(2*pi)*sum(F(t)^2*diag*t*dx);
end
    