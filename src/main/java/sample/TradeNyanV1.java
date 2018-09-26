package sample;

import mlnyan.ml.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;

public class TradeNyanV1 extends TradeNyanBase{

    LSTMCycle lstm;
    public TradeNyanV1(){
        PerceptronModel sig1 = new PerceptronModel(
                new Layer(1,4,Activator.RELU,new Momentum(0.8)),
                new Layer(4,8,Activator.RELU,new Momentum(0.8)),
                new Layer(8,12,Activator.RELU,new Momentum(0.8)),
                new Layer(12,16,Activator.RELU,new Momentum(0.8)),
                new Layer(16,10,Activator.RELU,new Momentum(0.8)),
                new Layer(10,8,Activator.RELU,new Momentum(0.8)),
                new Layer(8,1,Activator.SIGMOID,new Momentum(0.8))
                );
        PerceptronModel sig2 = new PerceptronModel(
                new Layer(1,4,Activator.RELU,new Momentum(0.8)),
                new Layer(4,8,Activator.RELU,new Momentum(0.8)),
                new Layer(8,12,Activator.RELU,new Momentum(0.8)),
                new Layer(12,16,Activator.RELU,new Momentum(0.8)),
                new Layer(16,10,Activator.RELU,new Momentum(0.8)),
                new Layer(10,8,Activator.RELU,new Momentum(0.8)),
                new Layer(8,1,Activator.SIGMOID,new Momentum(0.8))
        );PerceptronModel sig3 = new PerceptronModel(
                new Layer(1,4,Activator.RELU,new Momentum(0.8)),
                new Layer(4,8,Activator.RELU,new Momentum(0.8)),
                new Layer(8,12,Activator.RELU,new Momentum(0.8)),
                new Layer(12,16,Activator.RELU,new Momentum(0.8)),
                new Layer(16,10,Activator.RELU,new Momentum(0.8)),
                new Layer(10,8,Activator.RELU,new Momentum(0.8)),
                new Layer(8,1,Activator.SIGMOID,new Momentum(0.8))
        );PerceptronModel tanh = new PerceptronModel(
                new Layer(1,4,Activator.RELU,new Momentum(0.8)),
                new Layer(4,8,Activator.RELU,new Momentum(0.8)),
                new Layer(8,12,Activator.RELU,new Momentum(0.8)),
                new Layer(12,16,Activator.RELU,new Momentum(0.8)),
                new Layer(16,10,Activator.RELU,new Momentum(0.8)),
                new Layer(10,8,Activator.RELU,new Momentum(0.8)),
                new Layer(8,1,Activator.SIGMOID,new Momentum(0.8))
        );

        LSTMBlock block = new LSTMBlock(sig1,sig2,sig3,tanh,0.01);
        lstm = new LSTMCycle(block,0.01,
                new Layer(13,20,Activator.RELU,new Momentum(0.8)),
                new Layer(20,30,Activator.RELU,new Momentum(0.8)),
                new Layer(30,30,Activator.RELU,new Momentum(0.8)),
                new Layer(30,40,Activator.RELU,new Momentum(0.8)),
                new Layer(40,25,Activator.RELU,new Momentum(0.8)),
                new Layer(25,10,Activator.RELU,new Momentum(0.8)),
                new Layer(10,5,Activator.RELU,new Momentum(0.8))
                );
    }

    public void calc(String file){
        load(file);
        INDArray arr = Nd4j.create(new double[]{
                start.get(0),
                end.get(0),
                low.get(0),
                high.get(0),
                total.get(0),
                avarage5.get(0),
                avarage25.get(0),
                bollingerm3t.get(0),
                bollingerm2t.get(0),
                bollingerm1t.get(0),
                bollingerp1t.get(0),
                bollingerp2t.get(0),
                bollingerp3t.get(0)
        }).transpose();
        lstm.getCurrentModel().setInput(arr);
        System.out.println(lstm.getCurrentModel().calc());
    }

    public void learn(){
        File[] files = new File("./2018").listFiles();
        for (File file:files){
            if(file.getName().startsWith("."))continue;
            System.out.println(file.getName());
            load(file.getPath());
            INDArray[] bp = new INDArray[2];
            for (int i = 0;i < 2 /*start.size() - 1*/;++i){
                INDArray arr = Nd4j.create(new double[]{
                        start.get(i),
                        end.get(i),
                        low.get(i),
                        high.get(i),
                        total.get(i),
                        avarage5.get(i),
                        avarage25.get(i),
                        bollingerm3t.get(i),
                        bollingerm2t.get(i),
                        bollingerm1t.get(i),
                        bollingerp1t.get(i),
                        bollingerp2t.get(i),
                        bollingerp3t.get(i)
                }).transpose();
                lstm.getCurrentModel().setInput(arr);
                System.out.println(lstm.getCurrentModel().calc());
                lstm.next();
                INDArray arrbp = Nd4j.create(new double[]{
                        start.get(i + 1),
                        end.get(i + 1),
                        low.get(i + 1),
                        high.get(i + 1),
                        total.get(i + 1),
                });
                bp[i] = arrbp;
            }
            System.out.println(bp[0]);
            lstm.bp(bp);
        }
    }
}
