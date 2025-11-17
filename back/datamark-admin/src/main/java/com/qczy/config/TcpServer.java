package com.qczy.config;

/**
 * @author ：gwj
 * @date ：Created in 2024-10-29 14:53
 * @description：
 * @modified By：
 * @version: $
 */
import cn.hutool.core.util.ObjectUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.qczy.common.constant.BizConstants;
import com.qczy.handler.MyWebSocketHandler;
import com.qczy.mapper.ModelAssessConfigMapper;
import com.qczy.mapper.ModelAssessTaskMapper;
import com.qczy.model.entity.AlgorithmModelEntity;
import com.qczy.model.entity.AlgorithmTaskEntity;
import com.qczy.model.entity.ModelAssessTaskEntity;
import com.qczy.service.AlgorithmModelService;
import com.qczy.service.AlgorithmService;
import com.qczy.service.AlgorithmTaskService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import javax.annotation.PostConstruct;
import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.Date;

@Service
public class TcpServer {

    private static final int PORT = 9090; // TCP 服务监听的端口
    private static final int PORT1 = 9089; // TCP 服务监听的端口

    @Autowired
    MyWebSocketHandler myWebSocketHandler;


    @Autowired
    private AlgorithmTaskService algorithmTaskService;

    private AlgorithmTaskEntity algorithmTaskEntity;

    private AlgorithmTaskEntity algorithmTaskEntityException;

    @Autowired
    private AlgorithmModelService algorithmModelService;

    private AlgorithmModelEntity algorithmModelEntity;


    @PostConstruct
    public void startServer() {
        new Thread(() -> {
            try (ServerSocket serverSocket = new ServerSocket(PORT)) {
                System.out.println("TCP Server started on port " + PORT);

                while (true) {
                    Socket clientSocket = serverSocket.accept(); // 接收客户端连接
                    handleClient(clientSocket);
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }).start();
    }
    private void handleClient(Socket clientSocket) {
        new Thread(() -> {
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()));
            ) {
                String inputLine;
                //读取第一行作为任务id

                inputLine = reader.readLine();

                algorithmTaskEntity = algorithmTaskService.getOne(new LambdaQueryWrapper<AlgorithmTaskEntity>()
                        .eq(AlgorithmTaskEntity::getTaskId, inputLine));


                while ((inputLine = reader.readLine()) != null) {
                    System.out.println("Received from client: " + inputLine);
                    myWebSocketHandler.sendMessageToUser(BizConstants.TERMINAL_PROGRESS, "1", inputLine);
                }
            } catch (Exception e) {
                algorithmTaskEntity.setTaskStat("异常");
                algorithmTaskService.editTaskInfo(algorithmTaskEntity);

                algorithmModelEntity = algorithmModelService.getOne(new LambdaQueryWrapper<AlgorithmModelEntity>()
                        .eq(AlgorithmModelEntity::getTrainTaskId, algorithmTaskEntity.getTaskId()));

                if(ObjectUtil.isNotEmpty(algorithmModelEntity)){
                    algorithmModelEntity.setTrainStat("失败");
                    algorithmModelService.updateById(algorithmModelEntity);
                }
                e.printStackTrace();
                try {
                    clientSocket.close();
                } catch (IOException ex) {
                    throw new RuntimeException(ex);
                }
                return;
            }
            try {
                algorithmTaskEntity.setTaskStat("结束(自动)");
                algorithmTaskEntity.setUpdateTime(new Date());
                algorithmTaskService.editTaskInfo(algorithmTaskEntity);

                algorithmModelEntity = algorithmModelService.getOne(new LambdaQueryWrapper<AlgorithmModelEntity>()
                        .eq(AlgorithmModelEntity::getTrainTaskId, algorithmTaskEntity.getTaskId()));
                if(ObjectUtil.isNotEmpty(algorithmModelEntity)){
                    algorithmModelEntity.setTrainStat("成功");
                    algorithmModelService.updateById(algorithmModelEntity);
                }
                clientSocket.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }).start();
    }

    @PostConstruct
    public void startServer1() {
        new Thread(() -> {
            try (ServerSocket serverSocket = new ServerSocket(PORT1)) {
                System.out.println("TCP Server started on port " + PORT1);

                while (true) {
                    Socket clientSocket = serverSocket.accept(); // 接收客户端连接
                    handleClient1(clientSocket);
                }
            } catch (Exception e) {
                algorithmTaskEntityException.setTaskStat("异常");
                e.printStackTrace();
            }
        }).start();
    }

    private void handleClient1(Socket clientSocket) {
        new Thread(() -> {
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()));
            ) {

                String inputLine;
                inputLine = reader.readLine();
                algorithmTaskEntityException = algorithmTaskService.getOne(new LambdaQueryWrapper<AlgorithmTaskEntity>()
                        .eq(AlgorithmTaskEntity::getTaskId, inputLine));

                while ((inputLine = reader.readLine()) != null) {
                    System.out.println("Received from client: " + inputLine);
                    myWebSocketHandler.sendMessageToUser(BizConstants.EXCEPTION_TERMINAL_PROGRESS, "1", inputLine);
                }
            } catch (Exception e) {
                algorithmTaskEntityException.setTaskStat("异常");
                algorithmTaskService.editTaskInfo(algorithmTaskEntityException);
                e.printStackTrace();
            }

        }).start();
    }


}
