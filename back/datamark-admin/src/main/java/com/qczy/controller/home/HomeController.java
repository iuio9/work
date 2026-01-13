package com.qczy.controller.home;

import cn.hutool.core.bean.BeanUtil;
import cn.hutool.core.date.DateUtil;
import cn.hutool.core.map.MapUtil;
import cn.hutool.core.math.MathUtil;
import cn.hutool.core.util.NumberUtil;
import cn.hutool.core.util.ObjectUtil;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.fasterxml.jackson.annotation.JsonIdentityInfo;
import com.qczy.common.result.Result;
import com.qczy.mapper.ComputerInfoMapper;
import com.qczy.mapper.SystemStatusMapper;
import com.qczy.model.entity.ComputerInfoEntity;
import com.qczy.model.entity.SystemStatusEntity;
import com.qczy.model.entity.domain.Cpu;
import com.qczy.model.entity.domain.Info;
import com.qczy.model.entity.domain.Server;
import com.qczy.utils.*;
import io.swagger.annotations.Api;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.util.ObjectUtils;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import sun.net.www.http.HttpClient;

import java.io.UnsupportedEncodingException;
import java.net.URL;
import java.net.URLDecoder;
import java.net.URLEncoder;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;

import static org.apache.tomcat.jni.Address.getInfo;

@RestController
@RequestMapping("/home")
@Api(tags = "首页（算法系统）统计信息")
public class HomeController {

    @Autowired
    private HttpUtil httpUtil;

    @Value("${monitor.server}")
    private String server;

    @Autowired
    private ComputerInfoMapper computerInfoMapper;

    @Autowired
    private SystemStatusMapper systemStatusMapper;


    public static void main(String[] args) throws UnsupportedEncodingException {
        //	9.6447740350464E13	4.905019150336E12
        System.out.println(URLDecoder.decode("500kV%25E5%25AE%2589%25E8%2594%25A1%25E4%25B8%2580%25E7%25BA%25BF%25230087%25E5%25A1%2594-0086%25E5%25A1%2594-%25E9%2597%25B4%25E9%259A%2594%25E6%25A3%2592%25E5%25A4%25A7%25E5%2588%25B0%25E5%25B0%258F%25E5%258F%25B3%25E7%259B%25B8%25E7%25AC%25AC8%25E4%25B8%25AA%25E9%2597%25B4%25E9%259A%2594%25E6%25A3%2592%25E9%2594%2580%25E9%2592%2589%25E6%259C%25AA%25E6%2589%2593%25E5%25BC%25802-DJI-2920_dense.jpg", "utf-8"));
        System.out.println(URLUtils.encodeURL("500kV%E5%AE%89%E8%94%A1%E4%B8%80%E7%BA%BF%230087%E5%A1%94-0086%E5%A1%94-%E9%97%B4%E9%9A%94%E6%A3%92%E5%A4%A7%E5%88%B0%E5%B0%8F%E5%8F%B3%E7%9B%B8%E7%AC%AC8%E4%B8%AA%E9%97%B4%E9%9A%94%E6%A3%92%E9%94%80%E9%92%89%E6%9C%AA%E6%89%93%E5%BC%802-DJI-2920_dense.jpg"));
    }


    @GetMapping("/getComputerInfo")
    public Result getComputerInfo() {
        // API接口请求
        // （组装请求路径）
        try {
            LocalDateTime startOfDay = LocalDate.now().atStartOfDay();
            LocalDateTime endOfDay = LocalDate.now().plusDays(1).atStartOfDay();
            String start = DateUtil.format(startOfDay, "yyyy-MM-dd HH:mm:ss");
            String end = DateUtil.format(endOfDay, "yyyy-MM-dd HH:mm:ss");


            //.ge(YourEntity::getCreateTime, startOfDay)  // 当天开始时间
            //            .lt(YourEntity::getCreateTime, endOfDay)    // 第二天开始时间

            SystemStatusEntity algorithmInfo = systemStatusMapper.selectOne(
                    new LambdaQueryWrapper<SystemStatusEntity>()
                            .eq(SystemStatusEntity::getType,"0")
                            .ge(SystemStatusEntity::getCreateTime, start)  // 当天开始时间
                            .lt(SystemStatusEntity::getCreateTime, end)
                            .orderByDesc(SystemStatusEntity::getCreateTime)
                            .last("limit 1"));

            SystemStatusEntity storeInfo = systemStatusMapper.selectOne(
                    new LambdaQueryWrapper<SystemStatusEntity>()
                            .eq(SystemStatusEntity::getType,"1")
                            .ge(SystemStatusEntity::getCreateTime, start)  // 当天开始时间
                            .lt(SystemStatusEntity::getCreateTime, end)
                            .orderByDesc(SystemStatusEntity::getCreateTime)
                            .last("limit 1"));

            HashMap storeMap = getInfo(storeInfo);
            HashMap algorithmMap = getInfo(algorithmInfo);

            Map hashMap = new HashMap();

            hashMap.put("storeage",storeMap);
            hashMap.put("algorithm",algorithmMap);
            return Result.ok(hashMap);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return Result.ok();
    }

    /**
     * 创建默认的系统信息Map（当数据库中没有数据时使用）
     */
    private HashMap createDefaultInfoMap() {
        Info cpu = new Info();
        cpu.setTotal("100%");
        cpu.setFree("100%");
        cpu.setUsed("0%");
        cpu.setUsage("0%");

        Info gpu = new Info();
        gpu.setTotal("0MB");
        gpu.setFree("0MB");
        gpu.setUsed("0MB");
        gpu.setUsage("0%");

        Info mem = new Info();
        mem.setTotal("0B");
        mem.setFree("0B");
        mem.setUsed("0B");
        mem.setUsage("0%");

        Info disk = new Info();
        disk.setTotal("0B");
        disk.setFree("0B");
        disk.setUsed("0B");
        disk.setUsage("0%");

        HashMap serverMap = new HashMap();
        serverMap.put("cpu", cpu);
        serverMap.put("gpu", gpu);
        serverMap.put("sysFileinfo", disk);
        serverMap.put("mem", mem);
        return serverMap;
    }

    private HashMap getInfo(SystemStatusEntity systemStatusEntity) {
        // 添加空指针检查
        if (systemStatusEntity == null) {
            return createDefaultInfoMap();
        }

        Info cpu = new Info();
        cpu.setTotal("100%");
        if(ObjectUtil.isNotEmpty(systemStatusEntity.getCpuUsage())){
            cpu.setFree(100-Double.parseDouble(systemStatusEntity.getCpuUsage())+"%");
        }else{
            cpu.setFree("100%");
        }
//        cpu.setUsed(systemStatusEntity.getCpuUsed()+"ticks");
        if(ObjectUtil.isNotEmpty(systemStatusEntity.getCpuUsed())){
            cpu.setUsed(systemStatusEntity.getCpuUsage()+"%");
        }else {
            cpu.setUsed("0%");
        }
        if(ObjectUtil.isNotEmpty(systemStatusEntity.getCpuUsage())){
            cpu.setUsage(systemStatusEntity.getCpuUsage()+"%");
        }else {
            cpu.setUsage("0%");
        }



        Info gpu = new Info();
        gpu.setTotal(systemStatusEntity.getGpuTotal()+"MB");
        gpu.setFree(systemStatusEntity.getGpuFree()+"MB");
        gpu.setUsed(systemStatusEntity.getGpuUsed()+"MB");
        if(ObjectUtil.isNotEmpty(systemStatusEntity.getGpuUsage())){
            gpu.setUsage( systemStatusEntity.getGpuUsage()+"%");
        }else {
            gpu.setUsage("0%");
        }


        Info mem = new Info();
        mem.setTotal(FileFormatSizeUtils.formatSize(NumberUtil.toBigDecimal(systemStatusEntity.getMemTotal()).longValue()));
        mem.setFree(FileFormatSizeUtils.formatSize(NumberUtil.toBigDecimal(systemStatusEntity.getMemFree()).longValue()));
        mem.setUsed(FileFormatSizeUtils.formatSize(NumberUtil.toBigDecimal(systemStatusEntity.getMemUsed()).longValue()));
        mem.setUsage(FileFormatSizeUtils.calculatePercentage(NumberUtil.toBigDecimal(systemStatusEntity.getMemUsed()).longValue(),NumberUtil.toBigDecimal(systemStatusEntity.getMemTotal()).longValue()));


        Info disk = new Info();
        disk.setTotal(FileFormatSizeUtils.formatSize(NumberUtil.toBigDecimal(systemStatusEntity.getSysFileinfoTotal()).longValue()));
        disk.setFree(FileFormatSizeUtils.formatSize(NumberUtil.toBigDecimal(systemStatusEntity.getSysFileinfoFree()).longValue()));
        disk.setUsed(FileFormatSizeUtils.formatSize(NumberUtil.toBigDecimal(systemStatusEntity.getSysFileinfoUsed()).longValue()));
        disk.setUsage(FileFormatSizeUtils.calculatePercentage(NumberUtil.toBigDecimal(systemStatusEntity.getSysFileinfoUsed()).longValue(),NumberUtil.toBigDecimal(systemStatusEntity.getSysFileinfoTotal()).longValue()));

        HashMap serverMap = new HashMap();
        serverMap.put("cpu",cpu);
        serverMap.put("gpu",gpu);
        serverMap.put("sysFileinfo",disk);
        serverMap.put("mem",mem);
        return serverMap;
    }


    @GetMapping("/getDaysComputerInfo")
    public Result getDaysComputerInfo() {
        Map hashMap = new HashMap();
        hashMap.put("algorithmBar",systemStatusMapper.statisticalBar("0"));
        hashMap.put("storageBar", systemStatusMapper.statisticalBar("1"));

        hashMap.put("algorithmPie",systemStatusMapper.statisticalPie("0"));
        hashMap.put("storagePie", systemStatusMapper.statisticalPie("1"));
        return Result.ok(hashMap);
    }



}
