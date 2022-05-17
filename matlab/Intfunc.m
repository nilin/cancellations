function I=Intfunc(t,diag,F)

    x=log(t);
    dx=x(2)-x(1);
    integrand=(F(t).^2).*diag.*t.*dx;

    I=sqrt(2/pi)*sum(integrand,'all');
end
    