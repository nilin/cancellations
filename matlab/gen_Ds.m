mkdir('Ds')
for n=100:100
    mkdir(sprintf('Ds/n=%d',n));
    for i=1:100
        load(sprintf('Ws/n=%d/instance=%d',n,i));
        ts=exp(-10:.01:10);
        [diag,offdiag]=diagD(W,ts);
        save(sprintf('Ds/n=%d/instance=%d',n,i),'ts','diag','offdiag','W');
    end
    disp(n)
end