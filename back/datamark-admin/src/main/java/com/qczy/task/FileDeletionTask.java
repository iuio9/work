package com.qczy.task;

import com.qczy.mapper.TempFileMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.io.File;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/3/10 9:57
 * @Description:
 */
@Component
public class FileDeletionTask {

    private static final Logger log = LoggerFactory.getLogger(FileDeletionTask.class);

    @Value("${upload.tempPath}")
    private String tempPath;

    @Autowired
    private TempFileMapper tempFileMapper;

    // --------------- 删除临时文件夹的所有文件 ---------------
    @Scheduled(cron = "0 0 0 * * ?") // 每天执行一次
    //@Scheduled(cron = "0 * * * * ?")  // 每分钟执行一次
    public void deleteFiles() {
        log.info("---开始执行删除文件夹的定时任务---");
        File folder = new File(tempPath);
        if (folder.exists() && folder.isDirectory()) {
            File[] files = folder.listFiles();
            if (files != null) {
                for (File file : files) {
                    if (file.isFile()) {
                        file.delete();
                    }
                }
                log.info("---定时执行文件删除成功---");
            }
        }
    }


    // --------------- 清空临时文件表的所有数据 ---------------
    @Scheduled(cron = "0 0 0 * * ?") // 每天执行一次
     //@Scheduled(cron = "0 * * * * ?")  // 每分钟执行一次
    public void deleteMysqlFileData() {
        log.info("---开始执行删除临时数据的定时任务---");
        tempFileMapper.deleteFileData();
        log.info("---定时执行删除临时数据成功---");
    }


}
