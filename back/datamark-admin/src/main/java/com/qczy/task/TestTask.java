package com.qczy.task;

import com.qczy.handler.MyWebSocketHandler;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/13 15:44
 * @Description:
 */
@Component
public class TestTask {

    @Autowired
    private MyWebSocketHandler myWebSocketHandler;

/*    @Scheduled(cron = "0/5 * *  * * ? ")   //每5秒执行一次
    public void execute() {
       *//* SimpleDateFormat df = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss"); //设置日期格式
        System.out.println("欢迎访问 天津数据标注平台 " + df.format(new Date()));*//*
        myWebSocketHandler.sendMessageToUser("dataSetProgress","1","success");
    }*/
}
