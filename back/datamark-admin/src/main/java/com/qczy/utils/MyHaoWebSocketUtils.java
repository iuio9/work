package com.qczy.utils;

import cn.hutool.core.util.NumberUtil;
import com.alibaba.fastjson.JSONObject;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.qczy.common.result.Result;
import com.qczy.handler.MyWebSocketHandler;
import com.qczy.mapper.DataSonMapper;
import com.qczy.model.entity.DataSonEntity;
import com.qczy.service.impl.FileServiceImpl;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.math.BigDecimal;
import java.util.Objects;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/10/9 17:08
 * @Description:
 */
@Component
public class MyHaoWebSocketUtils {

    private static final Logger logger = LoggerFactory.getLogger(MyHaoWebSocketUtils.class);

    private static final int SUCCESS_CODE = 200;
    private static final int MAX_RETRIES = 3;
    private static final long RETRY_DELAY = 100; // 重试间隔时间，单位：毫秒

    @Autowired
    private MyWebSocketHandler myWebSocketHandler;

    @Autowired
    private DataSonMapper dataSonMapper;

    private static final Object lock = new Object();
    /**
     * @param sonId          数据集id
     * @param fileTotalNum   文件总数量
     * @param fileCurrentNum 当前总数量
     */
    public void sendMessage(String sonId, Integer fileTotalNum, Integer fileCurrentNum) {
        try {
            if (Objects.isNull(sonId) || Objects.isNull(fileTotalNum) || Objects.isNull(fileCurrentNum) || fileTotalNum == 0) {
                logger.error("输入参数异常，sonId: {}, fileTotalNum: {}, fileCurrentNum: {}", sonId, fileTotalNum, fileCurrentNum);
                return;
            }

            // 计算进度
            int progress = NumberUtil.div(fileCurrentNum.toString(), fileTotalNum.toString(), 2)
                    .multiply(BigDecimal.valueOf(100)).intValue();

            Result<Object> result = new Result<>();
            result.setCode(SUCCESS_CODE);
            result.setData(progress);

            synchronized (lock) {
                myWebSocketHandler.sendMessageToUser("dataSetProgress", sonId, JSONObject.toJSONString(result));
            }
            logger.info("已向用户 {} 发送进度信息，进度: {}%", sonId, progress);

            // 进度到100，关闭连接
            if (progress == 100) {
                myWebSocketHandler.disconnectUser("dataSetProgress", sonId);
                logger.info("已断开与用户 {} 的 WebSocket 连接", sonId);

                try {
                    DataSonEntity dataSonEntity = dataSonMapper.selectOne(new LambdaQueryWrapper<DataSonEntity>().eq(DataSonEntity::getSonId, sonId));
                    if (Objects.nonNull(dataSonEntity)) {
                        dataSonEntity.setIsSocket(0);
                        dataSonMapper.updateById(dataSonEntity);
                        logger.info("已更新用户 {} 的 isSocket 字段为 0", sonId);
                    } else {
                        logger.error("未找到数据集 ID 为 {} 的记录", sonId);
                    }
                } catch (Exception dbException) {
                    logger.error("更新数据库时出现异常，数据集 ID: {}", sonId, dbException);
                }
            }
        } catch (Exception e) {
            logger.error("发送消息时出现异常，数据集 ID: {}", sonId, e);
        }
    }


    /**
     * 发送普通文本消息
     *
     * @param taskId 任务ID
     * @param messageType 消息类型
     * @param text 文本内容
     */
    public void sendTextMessage(String taskId, String messageType, String text) {
        try {
            if (Objects.isNull(taskId) || Objects.isNull(messageType) || Objects.isNull(text)) {
                logger.error("发送文本消息参数异常，taskId: {}, messageType: {}, text: {}", taskId, messageType, text);
                return;
            }

            Result<Object> result = new Result<>();
            result.setCode(SUCCESS_CODE);
            result.setData(text);

            synchronized (lock) {
                myWebSocketHandler.sendMessageToUser(messageType, taskId, JSONObject.toJSONString(result));
            }

            logger.info("已发送类型为 {} 的文本消息到任务 {}", messageType, taskId);
        } catch (Exception e) {
            logger.error("发送文本消息时出现异常，任务ID: {}, 消息类型: {}", taskId, messageType, e);
        }
    }



    /**
     * @param sonId          数据集id
     * @param fileTotalNum   文件总数量
     * @param fileCurrentNum 当前总数量
     */
    /*public void sendMessage(String sonId, Integer fileTotalNum, Integer fileCurrentNum) {
        try {
            // 计算进度

            int progress = NumberUtil.div(fileCurrentNum, fileTotalNum, 2).multiply(BigDecimal.valueOf(100)).intValue();

            Result<Object> result = new Result<>();
            result.setCode(200);
            result.setData(progress);


            myWebSocketHandler.sendMessageToUser("dataSetProgress", sonId, JSONObject.toJSONString(result));

            // 进度到100，关闭连接
            if (progress == 100) {
                myWebSocketHandler.disconnectUser("dataSetProgress",sonId);
                DataSonEntity dataSonEntity = dataSonMapper.selectOne(new LambdaQueryWrapper<DataSonEntity>().eq(DataSonEntity::getSonId, sonId));
                dataSonEntity.setIsSocket(0);
                dataSonMapper.updateById(dataSonEntity);
            }


        } catch (Exception e) {
            e.printStackTrace();
        }
    }*/


    /**
     * @param sonId          数据集id
     * @param progress       进度
     */
    public void sendMessage(String sonId, Integer progress) {
        try {
            // 计算进度
            Result<Object> result = new Result<>();
            result.setCode(200);
            result.setData(progress);
            myWebSocketHandler.sendMessageToUser("dataSetProgress", sonId, JSONObject.toJSONString(result));

            // 进度到100，关闭连接
            if (progress == 100) {
                myWebSocketHandler.disconnectUser("dataSetProgress",sonId);
                DataSonEntity dataSonEntity = dataSonMapper.selectOne(new LambdaQueryWrapper<DataSonEntity>().eq(DataSonEntity::getSonId, sonId));
                if (dataSonEntity!=null){
                    dataSonEntity.setIsSocket(0);
                    dataSonMapper.updateById(dataSonEntity);
                }

            }


        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    public static void main(String[] args) {
        int i = (10 / 2) * 100;
        System.out.println(i);
    }

}
