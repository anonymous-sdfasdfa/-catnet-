function lbp = get_batch_lbp(Im) 
    lbp = [];
    for i = 1:8
        for j = 1:8
            batch = Im((i-1)*8+1:i*8,(j-1)*8+1:j*8);
            b_lbp = extractLBPFeatures(batch);
            lbp = [lbp,b_lbp];   
        end
    end