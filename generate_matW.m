%%
L = 40;
W = zeros(40,40,40);
for k = 1:40
    per = k/L;
    W(:,:,k) = generateW(L, per);
end
save('matW.mat','W');
