function [weight]=ComputeDistancFromBordereWeight(patch)
    %patch is binary patch, formulla of weight is 20*exp(-d^2/(2*3^2))
    weight=zeros(size(patch));
    
    %extract border
    SE=ones(3);
    border=patch-imerode(patch,SE);
    distance=0;
    delta=5;
    while(distance<500)
        weight(weight==0 & border==1)=3*exp(-distance^2/(2*delta^2));
        border=imdilate(border,SE);        
        distance=distance+1;
    end
    weight=weight+1;

end