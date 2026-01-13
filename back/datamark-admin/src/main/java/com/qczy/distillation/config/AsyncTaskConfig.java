package com.qczy.distillation.config;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;

import java.util.concurrent.Executor;
import java.util.concurrent.ThreadPoolExecutor;

/**
 * 异步任务配置
 *
 * 用于训练任务的异步执行
 * 注意：@EnableAsync已在MyApplication主类中启用，此处不重复声明
 *
 * @author AI Assistant
 * @date 2025-01-25
 */
@Configuration
public class AsyncTaskConfig {

    private static final Logger logger = LoggerFactory.getLogger(AsyncTaskConfig.class);

    /**
     * 创建训练任务专用的线程池
     *
     * @return 线程池执行器
     */
    @Bean(name = "taskExecutor")
    public Executor taskExecutor() {
        logger.info("初始化训练任务线程池...");

        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();

        // 核心线程数：同时运行的最大训练任务数
        int corePoolSize = 2;
        executor.setCorePoolSize(corePoolSize);

        // 最大线程数
        int maxPoolSize = 5;
        executor.setMaxPoolSize(maxPoolSize);

        // 队列容量：等待执行的任务队列大小
        int queueCapacity = 10;
        executor.setQueueCapacity(queueCapacity);

        // 线程名称前缀
        executor.setThreadNamePrefix("TrainingTask-");

        // 线程空闲时间（秒）
        executor.setKeepAliveSeconds(60);

        // 拒绝策略：队列满时，由调用线程执行
        executor.setRejectedExecutionHandler(new ThreadPoolExecutor.CallerRunsPolicy());

        // 等待所有任务完成后再关闭线程池
        executor.setWaitForTasksToCompleteOnShutdown(true);

        // 等待时间（秒）
        executor.setAwaitTerminationSeconds(60);

        executor.initialize();

        logger.info("训练任务线程池初始化完成: corePoolSize={}, maxPoolSize={}, queueCapacity={}",
                corePoolSize, maxPoolSize, queueCapacity);

        return executor;
    }
}
