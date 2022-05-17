d=3
mkdir('Ws')
for n=1:100
    mkdir(sprintf('Ws/n=%d',n));
    for i=1:100
        W=randn(n,d)/sqrt(n*d);
        fn=sprintf('Ws/n=%d/instance=%d',n,i);
        disp(fn)
        save(fn,'W');
    end
end