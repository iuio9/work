package com.qczy.task;

import cn.hutool.core.bean.BeanUtil;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.qczy.controller.home.HomeController;
import com.qczy.mapper.ComputerInfoMapper;
import com.qczy.mapper.SystemStatusMapper;
import com.qczy.model.entity.SystemStatusEntity;
import com.qczy.model.entity.domain.ComputerInfo;
import com.qczy.model.entity.domain.Info;
import com.qczy.model.entity.domain.Server;
import com.qczy.model.entity.domain.Sys;
import com.qczy.service.impl.HomeService;
import com.qczy.utils.HttpUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import org.springframework.util.ObjectUtils;

@Component
public class ComputerInfoTask {


    private static final Logger log = LoggerFactory.getLogger(ComputerInfoTask.class);

    @Autowired
    private HttpUtil httpUtil;

    @Value("${monitor.server}")
    private String server;

    @Autowired
    private ComputerInfoMapper computerInfoMapper;

    @Autowired
    private SystemStatusMapper systemStatusMapper;




    //@Scheduled(fixedRate = 5000)
    //@Scheduled(cron = "0 0 */2 * * ?") //每2个小时执行一次
  /*  @Scheduled(cron = "0/5 * *  * * ? ")
    public void addAlgorithmComputerTask() {
        log.info("--------------------->定时任务开始执行！");
        // 调用API接口
        String urlString = server; // 替换为你的API接口URL
        try {
            Object resultJson = httpUtil.get(urlString);
            if (!ObjectUtils.isEmpty(resultJson)) {
                JSONObject jsonObject1 = JSONUtil.parseObj(resultJson);
                Info gpu = BeanUtil.toBean(jsonObject1.get("gpu"), Info.class);
                Info cpu = BeanUtil.toBean(jsonObject1.get("cpu"), Info.class);
                Info mem = BeanUtil.toBean(jsonObject1.get("mem"), Info.class);
                Info sysFileInfo = BeanUtil.toBean(jsonObject1.get("sysFileInfo"), Info.class);
                ComputerInfo sys =  BeanUtil.toBean(jsonObject1.get("sys"), ComputerInfo.class);
                SystemStatusEntity systemStatusEntity = new SystemStatusEntity("0",gpu.total, gpu.used, gpu.free, gpu.usage,
                        cpu.total, cpu.used, cpu.free, cpu.usage,
                        mem.total, mem.used, mem.free, mem.usage,
                        sys.computerName, sys.computerIp, sys.userDir, sys.osName, sys.osArch,
                        sysFileInfo.total, sysFileInfo.used, sysFileInfo.free, sysFileInfo.usage
                );
                systemStatusMapper.insert(systemStatusEntity);
            }
        } catch (Exception e) {
            log.info("算法服务器首页数据入库失败");
        }

    }*/

    @Autowired
    HomeService homeService;
   /* @Scheduled(cron = "0/5 * *  * * ? ")
    public void addStoreageComputerTask() {
        log.info("--------------------->定时任务开始执行！");
        // 调用API接口
        try {
            Server storeageInfo = homeService.getStoreageInfo();
            if (!ObjectUtils.isEmpty(storeageInfo)) {
                JSONObject jsonObject1 = JSONUtil.parseObj(storeageInfo);
                Info gpu = BeanUtil.toBean(jsonObject1.get("gpu"), Info.class);
                Info cpu = BeanUtil.toBean(jsonObject1.get("cpu"), Info.class);
                Info mem = BeanUtil.toBean(jsonObject1.get("mem"), Info.class);
                Info sysFileInfo = BeanUtil.toBean(jsonObject1.get("sysFileInfo"), Info.class);
                ComputerInfo sys =  BeanUtil.toBean(jsonObject1.get("sys"), ComputerInfo.class);
                SystemStatusEntity systemStatusEntity = new SystemStatusEntity("1",gpu.total, gpu.used, gpu.free, gpu.usage,
                        cpu.total, cpu.used, cpu.free, cpu.usage,
                        mem.total, mem.used, mem.free, mem.usage,
                        sys.computerName, sys.computerIp, sys.userDir, sys.osName, sys.osArch,
                        sysFileInfo.total, sysFileInfo.used, sysFileInfo.free, sysFileInfo.usage
                );
                systemStatusMapper.insert(systemStatusEntity);
            }
        } catch (Exception e) {
            log.info("存储服务器首页数据入库失败");
        }

    }*/
}
