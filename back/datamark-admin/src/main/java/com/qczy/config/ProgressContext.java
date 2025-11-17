package com.qczy.config;

import com.qczy.task.ProgressListener;

/**
 * @author ：gwj
 * @date ：Created in 2024-08-24 15:43
 * @description：2.创建 ThreadLocal 来存储进度监听器
 * @modified By：
 * @version: $
 */
public class ProgressContext {
    private static final ThreadLocal<ProgressListener> progressListener = new ThreadLocal<>();

    public static void setProgressListener(ProgressListener listener) {
        progressListener.set(listener);
    }

    public static ProgressListener getProgressListener() {
        return progressListener.get();
    }

    public static void clear() {
        progressListener.remove();
    }
}
