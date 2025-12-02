package com.qczy.distillation.config;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.scheduling.annotation.EnableAsync;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;

import java.util.concurrent.Executor;
import java.util.concurrent.ThreadPoolExecutor;

/**
 * 异步任务配置
 *
 * 用于训练任务的异步执行
 *
 * @author AI Assistant
 * @date 2025-01-25
 */
@Configuration
@EnableAsync
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
        executor.setCorePoolSize(2);

        // 最大线程数
        executor.setMaxPoolSize(5);

        // 队列容量：等待执行的任务队列大小
        executor.setQueueCapacity(10);

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
                executor.getCorePoolSize(), executor.getMaxPoolSize(), executor.getQueueCapacity());

        return executor;
    }
}
