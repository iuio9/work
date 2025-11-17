package com.qczy.utils;

import cn.hutool.core.bean.BeanUtil;
import cn.hutool.core.collection.CollectionUtil;
import cn.hutool.core.io.FileUtil;
import cn.hutool.core.map.MapUtil;
import cn.hutool.core.util.ObjectUtil;
import cn.hutool.json.JSONArray;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.qczy.common.annotation.MonitorProgress;
import com.qczy.common.constant.AssessConstants;
import com.qczy.common.constant.BizConstants;
import com.qczy.common.result.Result;
import com.qczy.config.ProgressContext;
import com.qczy.config.TcpServer;
import com.qczy.handler.MyWebSocketHandler;
import com.qczy.mapper.*;
import com.qczy.model.entity.*;
import com.qczy.model.request.AlgorithmParams;
import com.qczy.service.*;
import com.qczy.task.ProgressListener;
import io.swagger.models.auth.In;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Component;

import javax.validation.ConstraintViolation;
import javax.validation.Valid;
import java.io.*;
import java.lang.reflect.Array;
import java.net.*;
import java.nio.file.Files;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * @author ：gwj
 * @date ：Created in 2024-08-28 10:40
 * @description：
 * @modified By：
 * @version: $
 */
@Component
public class TrainUtil {
    private static final Logger log = LoggerFactory.getLogger(TrainUtil.class);
    @Autowired
    AlgorithmModelService algorithmModelService;
    @Autowired
    MarkInfoMapper markInfoMapper;
    @Autowired
    AlgorithmTaskService algorithmTaskService;
    @Autowired
    FileService fileService;
    @Autowired
    ModelUtil modelUtil;

    @Autowired
    AlgorithmTaskResultMapper algorithmTaskResultMapper;

    @Autowired
    AlgorithmMapper algorithmlMapper;

    @Autowired
    DataSonService dataSonService;

    @Autowired
    DataMarkService dataMarkService;

    @Autowired
    AlgorithmMapper algorithmMapper;

    @Autowired
     MyWebSocketHandler myWebSocketHandler;


    @Value("${upload.formalPath}")
    private String formalPath;
    @Value("${file.accessAddress}")
    private String accessAddress;
    @Value("${flask.algorithmPicHttp1}")
    private String algorithmPicHttp;
    @Value("${flask.prefixToRemove1}")
    private String prefixToRemove;
    @Autowired
    private ModelAssessTaskMapper modelAssessTaskMapper;

    public void getParams(){

    }

    public  String removeFileExtension(String filename) {
        if (filename == null || filename.isEmpty()) {
            return filename;
        }

        int lastDotIndex = filename.lastIndexOf(".");
        if (lastDotIndex == -1) {
            // 没有点，说明文件没有后缀
            return filename;
        }
        return filename.substring(0, lastDotIndex);
    }

    public void recordTaskParams( AlgorithmTaskEntity algorithmTaskEntity,String genResult){
        AlgorithmTaskResultEntity algorithmTaskResultEntity = new AlgorithmTaskResultEntity();
        algorithmTaskResultEntity.setTaskId(algorithmTaskEntity.getTaskId());
        algorithmTaskResultEntity.setTaskParams(JSONUtil.toJsonStr(algorithmTaskEntity.getParams()));
        algorithmTaskResultEntity.setTaskResult(JSONUtil.isJson(genResult)?JSONUtil.toJsonStr(genResult):genResult);
        algorithmTaskResultMapper.insert(algorithmTaskResultEntity);
    }

    public void uploadDataSet(AlgorithmTaskEntity algorithmTaskEntity){
        algorithmTaskEntity.setIsAssess(AssessConstants.DATASETDOWNLOAD_ING);
        algorithmTaskService.editTaskInfo(algorithmTaskEntity);
        String dataSetId = algorithmTaskEntity.getDataSetId();
        DataSonEntity one = dataSonService.getOne(new LambdaQueryWrapper<DataSonEntity>().eq(
                DataSonEntity::getSonId, dataSetId
        ));
        AlgorithmEntity algorithmEntity = algorithmMapper.selectById(11);

        //拿到数据集的目录
        String dataSetPath = formalPath +  one.getFatherId() + "/v" + one.getVersion();
        String dataSetHttpPath = accessAddress  +  one.getFatherId() + "/v" + one.getVersion();
        // 获取json文件夹中的所有文件
        File jsonFolder = new File(dataSetPath + "/json/");
        File[] jsonFilesArray = null;
        if (jsonFolder.exists() && jsonFolder.isDirectory()) {
            jsonFilesArray = jsonFolder.listFiles();
        }

        // 获取source文件夹中的所有文件
        File sourceFolder = new File(dataSetPath + "/source/");
        if (sourceFolder.exists() && sourceFolder.isDirectory()) {
            File[] sourceFilesArray = sourceFolder.listFiles();
            if (sourceFilesArray != null) {
                for (File file : sourceFilesArray) {
                    if (file.isFile()) {
                        for (File file1 : jsonFilesArray) {
                            if(file1.isFile()){
                                if (removeFileExtension(file.getName()).equals(removeFileExtension(file1.getName()))) {
                                    //放入map
                                    Map<String,Object> jsonAndImgMap = new HashMap<>();
                                    jsonAndImgMap.put("json_path", URLUtils.encodeURL(dataSetHttpPath + "/json/" + file1.getName()));
                                    jsonAndImgMap.put("img_path",  URLUtils.encodeURL(dataSetHttpPath + "/source/" + file.getName()));
                                    AlgorithmParams algorithmParams = new AlgorithmParams();
                                    algorithmParams.setParams(jsonAndImgMap);
                                    String model = null;
                                    try{
                                        model = modelUtil.model(algorithmEntity, algorithmParams);

                                    }catch (Exception e){
                                        recordTaskParams(algorithmTaskEntity,e.getMessage());
                                    }
                                    recordTaskParams(algorithmTaskEntity,model);
                                }
                            }
                        }
                    }
                }
            }
        }

    }

    public void process_front(AlgorithmTaskEntity algorithmTaskEntity){
        Map<String, Object> params = algorithmTaskEntity.getParams();
        AlgorithmParams algorithmParams = new AlgorithmParams();
        algorithmParams.setParams(params);
        AlgorithmEntity algorithmEntity = algorithmMapper.selectById(12);
        String model = null;
        try{
            model = modelUtil.model(algorithmEntity, algorithmParams);

        }catch (Exception e){
            recordTaskParams(algorithmTaskEntity,e.getMessage());
        }
        recordTaskParams(algorithmTaskEntity,model);
    }


    //此处算法会异步返回http(包含一个训练的http浏览器链接)接口 并通过tcp实时回报进度(黑窗口)
    public void train(AlgorithmTaskEntity algorithmTaskEntity){


        Map<String, Object> params = algorithmTaskEntity.getParams();
        JSONArray trainParams = JSONUtil.parseArray(params.get("trainPrams"));

        Map<String, Object> stringObjectHashMap = new HashMap<>();
        for (Object trainParam : trainParams) {
            Map bean = BeanUtil.toBean(trainParam, Map.class);
            stringObjectHashMap.put(bean.get("key").toString(),bean.get("value"));
        }
        stringObjectHashMap.put("index",algorithmTaskEntity.getTaskId());
        stringObjectHashMap.put("mode",params.get("mode"));
        AlgorithmParams algorithmParams = new AlgorithmParams();

        algorithmParams.setParams(stringObjectHashMap);
        AlgorithmEntity algorithmEntity = algorithmMapper.selectById(13);
        //训练会返回两个链接
        String model = null;
        try{
            model = modelUtil.model(algorithmEntity, algorithmParams);
        }catch (Exception e){
            recordTaskParams(algorithmTaskEntity,e.getMessage());
            return;
        }
        updateModelInfo(algorithmTaskEntity,algorithmEntity,model,algorithmParams);
        recordTaskParams(algorithmTaskEntity,model);

    }

    public void updateModelInfo(AlgorithmTaskEntity taskEntity,AlgorithmEntity algorithmEntity,String model,AlgorithmParams algorithmParams) {
        AlgorithmModelEntity byId = algorithmModelService.getById(taskEntity.getModelId());

        if(JSONUtil.isJson(model)){
            JSONObject jsonObject = JSONUtil.parseObj(model);

            if(ObjectUtil.isNotEmpty(jsonObject.get("pid"))){
                JSONArray pids =  JSONUtil.parseArray(jsonObject.get("pid"));
                StringBuffer sb = new StringBuffer();
                for (int i1 = 0; i1 < pids.size(); i1++) {
                    sb.append(Integer.parseInt(pids.get(i1).toString()));
                    if(pids.size() -1  == i1){
                        break;
                    }
                    sb.append(",");

                }

                //将pid入库
                taskEntity.setPid(sb.toString());
            }


            if(ObjectUtil.isNotEmpty(jsonObject.get("wandb"))){
                byId.setTrainUrl( jsonObject.get("wandb").toString());
                taskEntity.setTrainUrl(jsonObject.get("wandb").toString());
            }

            if(ObjectUtil.isNotEmpty(jsonObject.get("model"))){
                AlgorithmModelEntity modelEntity = new AlgorithmModelEntity();
                modelEntity.setModelName(algorithmParams.getParams().get("name").toString());
                modelEntity.setModelUrl(jsonObject.get("model").toString());
                modelEntity.setTrainStat("训练中");
                modelEntity.setTrainTaskId(taskEntity.getTaskId().toString());
                List<AlgorithmModelEntity> list = algorithmModelService.list(new LambdaQueryWrapper<AlgorithmModelEntity>().eq(
                        AlgorithmModelEntity::getModelName, modelEntity.getModelName().toString()
                ));

                if(list.size() == 0){
                    algorithmModelService.addModelInfo(modelEntity);
                }else{
                    algorithmModelService.updateModelInfo(modelEntity);
                }
            }
            algorithmTaskService.updateById(taskEntity);
        }
    }

    @Async
    @MonitorProgress
    public void execAssess(AlgorithmTaskEntity algorithmTaskEntity){
        algorithmTaskEntity.setTaskStat("进行中");
        algorithmTaskService.editTaskInfo(algorithmTaskEntity);
        //1. 提交数据集

        try{
            System.out.println("提交数据集开始");
            uploadDataSet(algorithmTaskEntity);
            System.out.println("提交数据集完成");
        }catch (Exception e){
            System.out.println("提交数据集报错");
            algorithmTaskEntity.setIsAssess(AssessConstants.DATASETDOWNLOAD_ERROR);
            algorithmTaskEntity.setTaskStat("异常");
            algorithmTaskService.editTaskInfo(algorithmTaskEntity);
            recordTaskParams(algorithmTaskEntity,e.getMessage());
            return;
        }
        //2 . 开始评估
        try{
            System.out.println("评估开始");
            assess(algorithmTaskEntity);
            System.out.println("评估完成");
        }catch (Exception e){
            log.info(e.getMessage());
            System.out.println("评估报错");
            recordTaskParams(algorithmTaskEntity,e.getMessage());

            algorithmTaskEntity.setIsAssess(AssessConstants.ASSESS_ERROR);
            algorithmTaskEntity.setTaskStat("异常");
            algorithmTaskService.editTaskInfo(algorithmTaskEntity);
            return;
        }
        algorithmTaskEntity.setTaskStat("结束");
        algorithmTaskService.editTaskInfo(algorithmTaskEntity);

    }

    @Autowired
    private ModelAssessConfigMapper modelAssessConfigMapper;




    //关联评估
    private void associationAssess(AlgorithmTaskEntity algorithmTaskEntity){

        String modelId = algorithmTaskEntity.getModelId();
        AlgorithmModelEntity model = algorithmModelService.getOne(new LambdaQueryWrapper<AlgorithmModelEntity>()
                .eq(AlgorithmModelEntity::getModelId, modelId));

        if(ObjectUtil.isEmpty(model)){
            return;
        }
        if(ObjectUtil.isEmpty(model.getAssessLst())){
            model.setAssessLst(algorithmTaskEntity.getTaskId()+"");
        }else{
            model.setAssessLst(model.getAssessLst()+","+ algorithmTaskEntity.getTaskId()+"");
        }
        algorithmModelService.updateById(model);
    }

    //压缩当前数据集下的文件
    public String zipFile(AlgorithmTaskEntity algorithmTaskEntity){
        return "";
    }

    public void assess(AlgorithmTaskEntity algorithmTaskEntity) {
        System.out.println("开始评估");
        associationAssess(algorithmTaskEntity);
        Map<String,Object> assessParams = new HashMap();
//        AlgorithmModelEntity byId = algorithmModelService.getById(algorithmTaskEntity.getModelId());
        assessParams.put("weight",algorithmTaskEntity.getModelId());
//        assessParams.put("name",algorithmTaskEntity.getTrainType());
        AlgorithmParams algorithmParams = new AlgorithmParams();
        algorithmParams.setParams(assessParams);
        AlgorithmEntity algorithmEntity = algorithmMapper.selectById(15);
        System.out.println("调用评估");
        String model = modelUtil.model(algorithmEntity, algorithmParams);
        System.out.println("调用结果"+JSONUtil.toJsonStr(model));
        String dataBaseParams = algorithmEntity.getResponseParams();
        //解析params 从数据结中匹配
        if(ObjectUtil.isEmpty(dataBaseParams) || !JSONUtil.isJson(model)){
            return;
        }
        JSONArray jsonArray = JSONUtil.parseArray(dataBaseParams);
        List<Map> dataBaseResponseParamsMap = JSONUtil.toList(jsonArray, Map.class);
        algorithmTaskEntity.setIsAssess(AssessConstants.ASSESS_ED);
        //将model 保存到 assessUrl
        for (Map responseMap : dataBaseResponseParamsMap) {
            String key = responseMap.get("serverKey").toString();
            if (responseMap.get("type").equals("path")) {
                String fileDirOrName = "";
                try {
                    fileDirOrName = JSONUtil.parseObj(model).get(key).toString();
                } catch (Exception e) {
                    continue;
                }
                fileDirOrName = URLUtils.encodeURL(algorithmPicHttp + fileDirOrName.replaceFirst("^" + prefixToRemove, ""));
                String newFilePath = formalPath +  "/assess/" + algorithmTaskEntity.getTaskId() +  "/";
                try {

                    String s = downloadFile(fileDirOrName, newFilePath);

                    String assessUrl = URLUtils.encodeURL(accessAddress + s.replaceFirst("^" + formalPath, ""));

                    if (ObjectUtil.isNotEmpty(algorithmTaskEntity.getAssessUrl())) {
                        algorithmTaskEntity.setAssessUrl(algorithmTaskEntity.getAssessUrl() + ","+ assessUrl);
                    }else{
                        algorithmTaskEntity.setAssessUrl(assessUrl);
                    }
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
        }

        algorithmTaskService.editTaskInfo(algorithmTaskEntity);

        recordTaskParams(algorithmTaskEntity,model);

    }


    public void thirdCheckAssess(ModelAssessTaskEntity modelAssessTaskEntity) {
        thirdAssess(modelAssessTaskEntity,29,"thirdAssess");
    }
    public void thirdClassifyAssess(ModelAssessTaskEntity modelAssessTaskEntity) {
        thirdAssess(modelAssessTaskEntity,33,"thirdClassifyAssess");
    }
    public void thirdAssess(ModelAssessTaskEntity modelAssessTaskEntity,Integer type,String resultPath) {
        Map<String,Object> assessParams = new HashMap();
//        AlgorithmModelEntity byId = algorithmModelService.getById(algorithmTaskEntity.getModelId());
        assessParams.put("start",modelAssessTaskEntity.getId());
//        assessParams.put("name",algorithmTaskEntity.getTrainType());
        AlgorithmParams algorithmParams = new AlgorithmParams();
        algorithmParams.setParams(assessParams);
        AlgorithmEntity algorithmEntity = algorithmMapper.selectById(type);
        String model = modelUtil.model(algorithmEntity, algorithmParams);
        System.out.println("算法给出model"+model);
        JSONObject jsonObject = JSONUtil.parseObj(model);

        String dataBaseParams = algorithmEntity.getResponseParams();
        //解析params 从数据结中匹配
        if(ObjectUtil.isEmpty(dataBaseParams) || !JSONUtil.isJson(model)){
            return;
        }
        JSONArray jsonArray = JSONUtil.parseArray(dataBaseParams);
        List<Map> dataBaseResponseParamsMap = JSONUtil.toList(jsonArray, Map.class);
        //将model 保存到 assessUrl
        for (Map responseMap : dataBaseResponseParamsMap) {
            String key = responseMap.get("serverKey").toString();
            if (responseMap.get("type").equals("path")) {
                String fileDirOrName = "";
                try {
                    System.out.println("算法给我的文件地址"+fileDirOrName);
                    fileDirOrName = jsonObject.get(key).toString();
                } catch (Exception e) {
                    System.out.println("算法给我的文件地址报错了"+fileDirOrName);
                    continue;
                }
                fileDirOrName = URLUtils.encodeURL(algorithmPicHttp + fileDirOrName.replaceFirst("^" + prefixToRemove, ""));
                String newFilePath = formalPath + "/"+resultPath+"/" + modelAssessTaskEntity.getId() + "/";
                try {
                    System.out.println("下载编码后的文件"+fileDirOrName);
                    String s = downloadFile(fileDirOrName, newFilePath);
                    System.out.println("下载编码后的文件通过"+fileDirOrName);
                    String assessUrl = URLUtils.encodeURL(accessAddress + s.replaceFirst("^" + formalPath, ""));
                    jsonObject.set(key+"",assessUrl);
                    System.out.println("下载编码后的文件通过JSONUtil.parseObj(model).set(key+\"\",assessUrl)"+fileDirOrName);
                } catch (IOException e) {
//                    throw new RuntimeException(e);
                    System.out.println("下载编码后的文件异常"+e.getMessage());
                    jsonObject.set(key+"",fileDirOrName);
                }
            }
        }

//        algorithmTaskService.editTaskInfo(algorithmTaskEntity);
        System.out.println("结果为"+JSONUtil.toJsonStr(jsonObject));
        modelAssessTaskEntity.setTaskResult(JSONUtil.toJsonStr(jsonObject));
        modelAssessTaskMapper.updateById(modelAssessTaskEntity);
//        recordTaskParams(modelAssessTaskEntity,model);
    }

    public void sendMessage(String host, int port, String message) {
        try (Socket socket = new Socket(host, port);
             OutputStream outputStream = socket.getOutputStream();
             BufferedReader reader = new BufferedReader(new InputStreamReader(socket.getInputStream()))) {

            // 发送数据到服务器
            outputStream.write(message.getBytes());
            outputStream.flush();
            System.out.println("Message sent: " + message);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private Thread echoProcess(String msg){
        Thread thread = new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    myWebSocketHandler.sendMessageToUser(BizConstants.TERMINAL_PROGRESS, "1", msg);
                    Thread.sleep(2000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        });
        thread.start();
        return thread;

    }


    @Async
    @MonitorProgress
    public void execTask(AlgorithmTaskEntity algorithmTaskEntity){

        algorithmTaskEntity.setTaskStat("进行中");

        try{

            //1. 提交数据集
            Thread process = echoProcess("提交数据集中。。。。");
            uploadDataSet(algorithmTaskEntity);
            process.stop();

            //2. 调用预处理
            Thread process1 = echoProcess("调用预处理。。。。");
            process_front(algorithmTaskEntity);
            process1.stop();
            Thread process2 = echoProcess("调用训练模型。。。。");
            //3. 调用训练模型
            train(algorithmTaskEntity);
            process2.stop();
        }catch (Exception e){
            e.printStackTrace();
            recordTaskParams(algorithmTaskEntity,e.getMessage());
            algorithmTaskEntity.setTaskStat("异常");
            algorithmTaskService.editTaskInfo(algorithmTaskEntity);
        }

    }


    public static boolean isImageFile(String urlString) {
        try {

            URL url = new URL(urlString);
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("HEAD");
            String contentType = connection.getContentType();

            // 判断返回的Content-Type是否是图片类型
            return contentType.startsWith("image/");
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }


    public static boolean isDirectory(String urlString) {
        try {
//            String[] split = urlString.split("/");
//            split [split.length - 1]= URLEncoder.encode(split[split.length-1],"UTF-8");
//            urlString = ArrayUtil.join(split, "/");


//            // 原始 URL 的基础部分
//            String baseURL = "http://10.5.28.222/output_gsam_inpainting/";
//
//            // 中文路径部分
//            String chinesePath = "微信图片_20240906105012/";
//
//            // 对中文路径部分进行 URL 编码
//            String encodedChinesePath = URLEncoder.encode(chinesePath,"UTF-8");
//
//            // 恢复路径中的斜杠（URLEncoder 会对斜杠编码，需要恢复）
//            encodedChinesePath = encodedChinesePath.replace("%2F", "/");
//
//            // 拼接完整的 URL
//            String fullURL = baseURL + encodedChinesePath;
            URL url = new URL(urlString+"/");
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("HEAD");
            int responseCode = connection.getResponseCode();
            return responseCode == HttpURLConnection.HTTP_OK &&
                    (connection.getContentType() == null || connection.getContentType().startsWith("text/html"));
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    public static List<String> getFileURLs(String dirURL) throws IOException {
        List<String> fileURLs = new ArrayList<>();
        URL url = new URL(dirURL+"/");
        HttpURLConnection httpConn = (HttpURLConnection) url.openConnection();
        InputStream inputStream = httpConn.getInputStream();
        BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));

        String line;
        // 使用正则表达式来查找文件链接
        Pattern pattern = Pattern.compile("href=\"(.*?)\"");
        while ((line = reader.readLine()) != null) {
            Matcher matcher = pattern.matcher(line);
            while (matcher.find()) {
                String fileURL = matcher.group(1);
                if (!fileURL.endsWith("/")) { // 跳过子目录
                    fileURLs.add(dirURL + "/" +  fileURL);
                }
            }
        }
        reader.close();
        return fileURLs;
    }


    public static String downloadFile(String fileURL, String saveDir) throws IOException {
        URL url = new URL(fileURL);
        HttpURLConnection httpConn = (HttpURLConnection) url.openConnection();
        int responseCode = httpConn.getResponseCode();
        String saveFilePath = "";
        // 检查 HTTP 响应代码
        if (responseCode == HttpURLConnection.HTTP_OK) {

            String disposition = httpConn.getHeaderField("Content-Disposition");
            String fileName = "";
            if (disposition != null && disposition.contains("filename=")) {
                fileName =  URLDecoder.decode(disposition.substring(disposition.indexOf("filename=") + 9).replaceAll("\"", ""),"UTF-8");
            } else {
                // 从 URL 获取文件名
                fileName = URLDecoder.decode(fileURL.substring(fileURL.lastIndexOf("/") + 1),"UTF-8");
            }

            // 输入流从 HTTP 连接获取文件
            InputStream inputStream = httpConn.getInputStream();
            if(!FileUtil.exist(saveDir)){
                FileUtil.mkdir(saveDir);
            }
            saveFilePath = saveDir + "/" + URLDecoder.decode(fileName);

            // 打开输出流保存文件
            FileOutputStream outputStream = new FileOutputStream(saveFilePath);

            int bytesRead;
            byte[] buffer = new byte[4096];
            while ((bytesRead = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, bytesRead);
            }

            outputStream.close();
            inputStream.close();

            System.out.println("File downloaded: " + saveFilePath);
        } else {
            System.out.println("No file to download. Server replied HTTP code: " + responseCode);
        }
        httpConn.disconnect();
        return saveFilePath;
    }


    public static void main(String[] args) throws IOException {


//        String url ="http://192.168.1.4/results/20240927_141812/35KV%E6%A1%91%E7%AA%9D%E7%BA%BF016%23%E5%A1%94-%E4%B8%8A%E7%9B%B8%E5%B0%8F%E5%8F%B7%E4%BE%A7%E7%BB%9D%E7%BC%98%E5%AD%90%E6%B6%82%E5%B1%82%E7%A0%B4%E6%8D%9F-DSC-51320/";
//        String s = URLUtils.encodeURL(url);
//        System.out.println(s);
//        System.out.println(isDirectory(s));
//        List<String> fileURLs = getFileURLs(s);
//        for (String fileURL : fileURLs) {
//            System.out.println(downloadFile(fileURL,"."));
//        }
    }

    public static void test(){

        // 原始 URL 的基础部分
        String baseURL = "http://192.168.1.3:9092/formal//1288533794491465728/v1/source/35KV桑窝线016#塔-上相小号侧绝缘子涂层破损-DSC-51320.JPG";

        try {
            String fullPath = URLEncoder.encode(baseURL, "UTF-8").replaceAll("\\+", "%20")  // 替换空格
                    .replace("%3A", ":")       // 恢复冒号 ":"
                    .replace("%2F", "/");

            System.out.println(fullPath);
        } catch (UnsupportedEncodingException e) {
            throw new RuntimeException(e);
        }
    }


    public String trainStop(AlgorithmTaskEntity pid) {
        AlgorithmEntity algorithmEntity = algorithmMapper.selectById(14);
        AlgorithmParams algorithmParams = new AlgorithmParams();
        Map<String,Object> hashMap = new HashMap();
        hashMap.put("pid_list",pid.getPid());
        algorithmParams.setParams(hashMap);
        String model = modelUtil.model(algorithmEntity, algorithmParams);
        recordTaskParams(pid,model);
        pid.setTaskStat("结束(手动)");
        //结束训练时修改模型状态
        pid.setPid(null);
        AlgorithmModelEntity trainModel = algorithmModelService.getOne(new LambdaQueryWrapper<AlgorithmModelEntity>().eq(
                AlgorithmModelEntity::getTrainTaskId, pid.getTaskId()
        ));
        trainModel.setTrainStat("失败");
        algorithmModelService.updateById(trainModel);
        algorithmTaskService.editTaskInfo(pid);

        return model;
    }

    public HashMap getAssess(AlgorithmTaskEntity byId) {

        HashMap hashMap = new HashMap();
        hashMap.put("status","0");
        hashMap.put("result","正在评估中...");
        //需要一张图片和markInfo

        if(ObjectUtil.isEmpty(byId.getIsAssess())){
            byId.setIsAssess(AssessConstants.NOT_ASSESS);
        }
        if(byId.getIsAssess().equalsIgnoreCase(AssessConstants.ASSESS_ED)){
            hashMap.put("status","1");
            hashMap.put("result",byId.getAssessUrl());
            return hashMap;
        }else if(byId.getIsAssess().equalsIgnoreCase(AssessConstants.NOT_ASSESS)){
            byId.setIsAssess(AssessConstants.ASSESS_ING);
            return hashMap;
        }else if(byId.getIsAssess().equalsIgnoreCase(AssessConstants.DATASETDOWNLOAD_ING)){
            hashMap.put("result","数据集下载中...");
            return hashMap;
        }else if(byId.getIsAssess().equalsIgnoreCase(AssessConstants.DATASETDOWNLOAD_ERROR)){
            hashMap.put("result","下载数据集出错");
            return hashMap;
        }else if(byId.getIsAssess().equalsIgnoreCase(AssessConstants.ASSESS_ERROR)){
            hashMap.put("result","评估出错");
            return hashMap;
        }else if(byId.getIsAssess().equalsIgnoreCase(AssessConstants.ASSESS_ING)){
            return hashMap;
        }
        return hashMap;
    }
}
