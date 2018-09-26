package mlnyan.ml;

import java.util.ArrayList;

public class LSTMCycle{
    LSTMModel base;

    ArrayList<LSTMModel> models = new ArrayList<>();

    public LSTMCycle(LSTMBlock lstmBlock,Layer... layers){
        this.base = new LSTMModel(lstmBlock,layers);
        models.add(base);
    }

    public LSTMModel getCurrentModel(){
        return models.get(models.size() - 1);
    }

    public void next(){
        models.add(getCurrentModel().next());
    }

    public void bp(IND){

    }
}
