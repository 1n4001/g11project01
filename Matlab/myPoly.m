function f = myPoly (x,p)

f=0;

for i=1:(length(p))
    f = f + p(i)*x.^(length(p)-i);
end
