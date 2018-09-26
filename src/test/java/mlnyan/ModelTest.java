package mlnyan;

import mlnyan.ml.Activator;
import mlnyan.ml.Layer;
import mlnyan.ml.Model;
import mlnyan.ml.Momentum;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class ModelTest {

    Model model;

    @org.junit.Before
    public void setUp() throws Exception {
        model = new Model(new Momentum(0.01,0.7),new Layer(3,Activator.RELU),new Layer(8,Activator.SOFTMAX),new Layer(4,Activator.NONE));
        model.init();
    }

    @org.junit.After
    public void tearDown() throws Exception {
    }

    @org.junit.Test
    public void calc(){
        for (int i = 0;i < 1000;++i){
            model.setInput(Nd4j.create(new double[]{0.3323,0.4,0.7}).transpose());
            INDArray result =  model.calc();
            StringBuilder sb = new StringBuilder();
            for (double d:result.toDoubleVector()){
                sb.append(d);
                sb.append("/");
            }
            INDArray a = Nd4j.zeros(4,1);
            a.put(0,0,0.0);
            a.put(1,0,1);
            a.put(2,0,0.0);
            a.put(3,0,0.0);
            System.out.println(sb.toString());
            model.doBackPropagation(a);
        }
    }

    @org.junit.Test
    public void test(){
        INDArray a = Nd4j.zeros(5,1);
        System.out.println(a);
        a =  a.add(3);
        System.out.println(a);
    }


    @org.junit.Test
    public void convertToJson() {
        System.out.println(model.convertToJson());
    }
}