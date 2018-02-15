%%
L = 40;
W = zeros(40,40,10);
for k = 1:10
    per = (k+30)/L;
    W(:,:,k) = generateW(L, per);
end
save('matW.mat','W');
