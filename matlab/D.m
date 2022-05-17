function Dt=D(W,s,t)
    [n,d]=size(W);
    Wl=s*reshape(W,[n,1,d]);
	Wr=t*reshape(W,[1,n,d]);
	sqdist= sum((Wl-Wr).^2,3);
    Dt=det(exp(-(1/2)*sqdist));
