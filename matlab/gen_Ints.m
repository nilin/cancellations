mkdir('Ints')
mkdir('Ints/ReLU')
for n=1:100
    mkdir(sprintf('Ints/n=%d',n));
    %outs=zeros(100,1);
    for i=1:100
        load(sprintf('Ds/n=%d/instance=%d',n,i));
        outs(i)=Intfunc(ts,diag,@(x)1./(sqrt(2*pi)*x.^2))/sqrt(sum(W.^2,'all'));
    end
    disp(n);
    save(sprintf('Ints/ReLU/n=%d',n),'outs');
end