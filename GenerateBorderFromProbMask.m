function [res,mask]=GenerateBorderFromProbMask(org,probMask,thereshold)
    res=probMask;
    res1=GenerateMask(res,thereshold);
    res1= imfill(res1,'holes');
    
    [ym xm]=find(res==max(res(:)));
    ym=ym(1);xm=xm(1);
    while(1)
        resTemp=res1;
        resTemp(2:end-1,2:end-1)=0;
        resTemp=sum(resTemp(:));
        if( resTemp>0)
            indTemp=find(res1==1);
            res(indTemp)=0;
            res1=GenerateMask(res,0.5);
            
        else
            break;

        end
    end
    res1= imfill(res1);
%    SE=ones(10,10);
%    res1=imdilate(res1,SE);
    mask=res1;
    SE=ones(3,3);
    
    res=res1-imerode(res1,SE);
    SE=ones(3);
    res=imdilate(res,SE);
    r=org(:,:,1);
    r(res==1)=0;
    
    g=org(:,:,2);
    g(res==1)=0;
    b=org(:,:,3);
    b(res==1)=255;
    
    res=zeros(size(org));
    res(:,:,1)=r;
    res(:,:,2)=g;
    res(:,:,3)=b;
    
    
    
end