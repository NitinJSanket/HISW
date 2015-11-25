function [d] = histogramIntersection(w1, w2)
    d = sum(min([w1',w2'],[],2));
end