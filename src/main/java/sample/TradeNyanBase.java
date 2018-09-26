package sample;


import java.util.ArrayList;
import java.util.Comparator;

public abstract class TradeNyanBase{
    ArrayList<Double> start = new ArrayList<>();
    ArrayList<Double> end = new ArrayList<>();
    ArrayList<Double> high = new ArrayList<>();
    ArrayList<Double> low = new ArrayList<>();
    ArrayList<Double> avarage5 = new ArrayList<>();
    ArrayList<Double> avarage25 = new ArrayList<>();

    ArrayList<Double> bollingerm3t = new ArrayList<>();
    ArrayList<Double> bollingerm2t = new ArrayList<>();
    ArrayList<Double> bollingerm1t = new ArrayList<>();
    ArrayList<Double> bollingerp1t = new ArrayList<>();
    ArrayList<Double> bollingerp2t = new ArrayList<>();
    ArrayList<Double> bollingerp3t = new ArrayList<>();

    ArrayList<Double> total = new ArrayList<>();

    public static void main(String args[]){
        new TradeNyanV1().learn();
    }

    public void load(String filePath){
        CSV csv = CSV.loadForTradedata(filePath);
        ArrayList<String> startsa = csv.getArrayList("始値");
        ArrayList<Double> starta = new ArrayList<>();
        for (String s:startsa){
            starta.add(Double.valueOf(s));
        }

        ArrayList<String> endsa = csv.getArrayList("終値");
        ArrayList<Double> enda = new ArrayList<>();
        for (String s:endsa){
            enda.add(Double.valueOf(s));
        }

        ArrayList<String> highsa = csv.getArrayList("高値");
        ArrayList<Double> higha = new ArrayList<>();
        for (String s:highsa){
            higha.add(Double.valueOf(s));
        }

        ArrayList<String> lowsa = csv.getArrayList("安値");
        ArrayList<Double> lowa = new ArrayList<>();
        for (String s:lowsa){
            lowa.add(Double.valueOf(s));
        }

        ArrayList<String> totalsa = csv.getArrayList("出来高");
        ArrayList<Double> totala = new ArrayList<>();
        for (String s:totalsa){
            totala.add(Double.valueOf(s));
        }

        if(starta.size() < 25)return;
       /* int k = starta.size() - 24;
        start = new double[k];
        end = new double[k];
        high = new double[k];
        low = new double[k];
        total = new double[k];
        avarage5 = new double[k];
        avarage25 = new double[k];
        bollingerm3t = new double[k];
        bollingerm2t = new double[k];
        bollingerm1t = new double[k];
        bollingerp1t = new double[k];
        bollingerp2t = new double[k];
        bollingerp3t = new double[k];*/


        double a5 = 0;
        double a25 = 0;
        for (int i = 0;i < 20;++i){
            a25 += enda.get(i);
        }
        for (int i = 20;i < 24;++i){
            double d1 = enda.get(i);
            a5 += d1;
            a25 += d1;
        }

       /* double t = 0;
        double s = a25 / 25;
        for (int j = 0;j < 25;++j){
            double dd = Double.valueOf(enda.get(j)) - s;
            t += dd * dd;
        }
        t /= 25;
        t = Math.sqrt(t);*/

        for (int i = 24;i < starta.size();++i){
            double d1 = enda.get(i);
            double d2 = i == 24 ? 0 : enda.get(i - 5);
            double d3 = i == 24 ? 0 : enda.get(i - 25);
            a5 += d1 - d2;
            a25 += d1 - d3;
            double t = 0;
            avarage5.add(a5 / 25);
            avarage25.add(a25 / 25);
            for (int j = i - 24;j <= i;++j){
                double dd = enda.get(j) - avarage25.get(i - 24);
                t += dd * dd;
            }
            t = Math.sqrt(t / 25);
            start.add(starta.get(i));
            end.add(enda.get(i));
            high.add(higha.get(i));
            low.add(lowa.get(i));
            total.add(totala.get(i));
            bollingerp1t.add(avarage25.get(i - 24) + t);
            bollingerp2t.add(avarage25.get(i - 24) + t * 2);
            bollingerp3t.add(avarage25.get(i - 24) + t * 3);
            bollingerm1t.add(avarage25.get(i - 24) - t);
            bollingerm2t.add(avarage25.get(i - 24) - t * 2);
            bollingerm3t.add(avarage25.get(i - 24) - t * 3);
        }
/*
        ArrayList<Double> mh = new ArrayList<>(high);
        mh.sort((a,b) -> (int)(b - a));
        ArrayList<Double> mt = new ArrayList<>(total);
        mt.sort((a,b) -> (int)(b - a));
        ArrayList<Double> bl = new ArrayList<>(bollingerp3t);
        bl.sort((a,b) -> (int)(b - a));
        double maxh = mh.get(0);
        double maxt = mt.get(0);
        double bol = bl.get(0);
        for (int i = 0;i < start.size();++i){
            start.set(i,start.get(i) * 2 / maxh - 1);
            end.set(i,end.get(i) * 2 / maxh - 1);
            high.set(i,high.get(i) * 2 / maxh - 1);
            low.set(i,low.get(i) * 2 / maxh - 1);
            total.set(i,total.get(i) * 2 / maxt - 1);

            avarage5.set(i,avarage5.get(i) * 2 / maxh - 1);
            avarage25.set(i,avarage25.get(i) * 2 / maxh - 1);



            bollingerp1t.set(i,bollingerp1t.get(i) * 2 / bol - 1);
            bollingerp2t.set(i,bollingerp2t.get(i) * 2 / bol - 1);
            bollingerp3t.set(i,bollingerp3t.get(i) * 2 / bol - 1);
            bollingerm1t.set(i,bollingerm1t.get(i) * 2 / bol - 1);
            bollingerm2t.set(i,bollingerm2t.get(i) * 2 / bol - 1);
            bollingerm3t.set(i,bollingerm3t.get(i) * 2 / bol - 1);
        }*/
    }
}
