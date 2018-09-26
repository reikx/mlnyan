package mlnyan.sys;

import java.util.Calendar;

/**
 * System's Logger class.
 * @author Rei Kambayashi
 */
public class Logger {
    /**
     * Is logger swiched to DEBUG_MODE
     */
    private static boolean DEBUG_MODE = false;

    public static void setDebugMode(boolean debugMode) {
        DEBUG_MODE = debugMode;
    }

    /**
     * print debug message to System.out.
     * @param o Message or Object to print.
     */
    public static void debug(Object o){
        if(!DEBUG_MODE)return;
        StackTraceElement[] stes = new Throwable().getStackTrace();
        StringBuilder sb = new StringBuilder();
        sb.append("[DEBUG]");
        sb.append(getTime());
        sb.append('[');
        sb.append(stes[1].getClassName());
        sb.append(":");
        sb.append(stes[1].getLineNumber());
        sb.append(" => ");
        sb.append(stes[1].getMethodName());
        sb.append("()]");
        sb.append(':');
        sb.append(o);
        System.out.println(sb.toString());
    }

    /**
     * print error message to System.err.
     * @param o Message or Object to print.
     */
    public static void error(Object o){
        StringBuilder sb = new StringBuilder();
        sb.append("[ERROR]");
        sb.append(getTime());
        sb.append(':');
        sb.append(o);
        System.err.println(sb.toString());
    }

    /**
     * print warning message to System.out.
     * @param o Message or Object to print.
     */
    public static void warn(Object o){
        StringBuilder sb = new StringBuilder();
        sb.append("[WARN]");
        sb.append(getTime());
        sb.append(':');
        sb.append(o);
        System.out.println(sb.toString());
    }

    /**
     * print info message to System.out.
     * @param o Message or Object to print.
     */
    public static void info(Object o){
        StringBuilder sb = new StringBuilder();
        sb.append("[INFO]");
        sb.append(getTime());
        sb.append(':');
        sb.append(o);
        System.out.println(sb.toString());
    }

    /**
     * Get formatted current time.
     * @return formatted time
     */
    private static String getTime(){
        StringBuilder sb = new StringBuilder();
        Calendar calendar = Calendar.getInstance();
        sb.append('[');
        sb.append(calendar.get(Calendar.YEAR));
        sb.append('/');
        sb.append(normalize(calendar.get(Calendar.MONTH) + 1));
        sb.append('/');
        sb.append(normalize(calendar.get(Calendar.DATE)));
        sb.append(' ');
        sb.append(normalize(calendar.get(Calendar.HOUR_OF_DAY)));
        sb.append(':');
        sb.append(normalize(calendar.get(Calendar.MINUTE)));
        sb.append(':');
        sb.append(normalize(calendar.get(Calendar.SECOND)));
        sb.append(']');
        return sb.toString();
    }

    /**
     * Normalize parameter to two-digits.
     * @param i parameter to Normalize.
     * @return Normalized parameter.
     */
    private static String normalize(int i){
        return i < 10 ? "0" + i : String.valueOf(i);
    }
}
