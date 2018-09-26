package mlnyan.ml;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class MatrixUtil {
    public static INDArray sigmoidDeriv(INDArray src){
        INDArray out = Transforms.sigmoid(src);
        INDArray ia = Nd4j.ones(out.rows(),out.columns()).sub(out);
        return out.mul(ia);
    }

    public static INDArray tanhDeriv(INDArray src){
        INDArray out = Transforms.cosh(src);
        return Nd4j.ones(src.rows(),src.columns()).div(out.mul(out));
    }

    public static INDArray reluDeriv(INDArray src){
        INDArray out = Transforms.max(src,0);
        for (int row = 0;row < src.rows();++row){
            out.put(row,1,out.getDouble(row,1) > 0 ? 1 : 0);
        }
        return out;
    }

    public static INDArray softmaxDeriv(INDArray src){
        return sigmoidDeriv(src);
    }

}
