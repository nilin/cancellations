function [Ds,Ds_]=diagD(W,ts)
    Ds=zeros(size(ts,2),1);
    Ds_=zeros(size(ts,2),1);
    for i = 1:size(ts,2)
        t=ts(i);
        Ds(i)=D(W,t,t);
        Ds_(i)=D(W,t,-t);
    end
end