package com.qczy.utils;

import cn.hutool.core.date.DateUnit;
import cn.hutool.core.date.DateUtil;
import cn.hutool.core.date.LocalDateTimeUtil;
import cn.hutool.core.io.FileUtil;
import cn.hutool.core.map.MapUtil;
import cn.hutool.core.util.ArrayUtil;
import cn.hutool.core.util.ObjectUtil;
import cn.hutool.json.JSONArray;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.qczy.common.annotation.MonitorProgress;
import com.qczy.common.constant.AssessConstants;
import com.qczy.common.constant.BizConstants;
import com.qczy.config.ProgressContext;
import com.qczy.handler.MyWebSocketHandler;
import com.qczy.mapper.*;
import com.qczy.model.entity.*;

import com.qczy.model.request.AlgorithmParams;
import com.qczy.service.*;
import com.qczy.service.impl.AlgorithmServiceImpl;
import com.qczy.task.ProgressListener;
import io.swagger.models.auth.In;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.IOUtils;
import org.jpedal.parser.shape.B;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Component;

import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLDecoder;
import java.net.URLEncoder;
import java.nio.file.Files;
import java.sql.Array;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collector;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * @author ：gwj
 * @date ：Created in 2024-08-28 10:40
 * @description：
 * @modified By：
 * @version: $
 */
@Component
@Slf4j
public class TaskUtil {
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
    @Value("${flask.algorithmPicHttp}")
    private String algorithmPicHttp;

    @Value("${flask.algorithmPicHttp1}")
    private String algorithmPicHttp1;

    @Value("${flask.prefixToRemove}")
    private String prefixToRemove;

    @Value("${flask.prefixToRemove1}")
    private String prefixToRemove1;
    @Autowired
    private AlgorithmTaskMapper algorithmTaskMapper;
    @Autowired
    private AlgorithmServiceImpl algorithmService;
    @Autowired
    private HttpUtil httpUtil;

    @Async
    @MonitorProgress
    public void execTask(AlgorithmTaskEntity algorithmTaskEntity){
        int i = 0;
        String sonId = algorithmTaskEntity.getDataSetId();
        String modelId = algorithmTaskEntity.getModelId();
        Long taskId = algorithmTaskEntity.getTaskId();
        String algorithmId = algorithmTaskEntity.getAlgorithmId();
        System.out.println("-=================================================="+algorithmId);
        Map<String, Object> params = algorithmTaskEntity.getParams();
        LambdaQueryWrapper<DataSonEntity> queryWrapper = new LambdaQueryWrapper<DataSonEntity>().eq(DataSonEntity::getSonId, sonId);
        List<DataSonEntity> dataSonEntities = dataSonService.list(queryWrapper);
        String datasetOutId = algorithmTaskEntity.getDatasetOutId();


        algorithmTaskEntity.setTaskStat("进行中");

        algorithmTaskEntity.setVersion(dataSonEntities.get(0).getVersion()+"");
        algorithmTaskService.updateById(algorithmTaskEntity);
//        DataSonEntity dataSon = dataSonService.getOne(new LambdaQueryWrapper<DataSonEntity>().eq(DataSonEntity::getSonId, datasetOutId));

        AlgorithmEntity algorithmEntity = null;
        for (DataSonEntity dataSonEntity : dataSonEntities) {
            // 拼接正式上传文件的路径

            try{

                System.out.println("algorithmMapper==================="+algorithmMapper);
                algorithmEntity = algorithmMapper.selectOne(new LambdaQueryWrapper<AlgorithmEntity>()
//                            .eq(AlgorithmEntity::getModelId, algorithmTaskEntity.getModelId())
                                .eq(AlgorithmEntity::getId, algorithmTaskEntity.getAlgorithmId())
                );
                System.out.println("algorithmEntity==================="+algorithmEntity);
                if(algorithmEntity!=null){
                    algorithmEntity.setCurTaskId(taskId+"");
                    algorithmlMapper.updateById(algorithmEntity);
                }
                String newFilePath = formalPath + dataSonEntity.getFatherId() + "/" + "v" + dataSonEntity.getVersion() + "/" + "generate" +  "/" + algorithmEntity.getAlgorithmName()+"/" ;
                File fileMkdir = new File(newFilePath);
                // 判断目录是否存在，不存在，则创建
                if (!fileMkdir.exists()) {
                    fileMkdir.mkdirs();
                }


                AlgorithmParams algorithmParams = new AlgorithmParams();

                String fileIds = dataSonEntity.getFileIds();

                // 使用 Stream API 将字符串转换为 Integer
                List<Integer> integerList = Arrays.stream(fileIds.split(","))
                        .map(Integer::parseInt)  // 将每个字符串转换为 Integer
                        .collect(Collectors.toList());

                // 打印结果
                System.out.println(integerList);
                List<FileEntity> fileEntities = fileService.listByIds(integerList);


                for (FileEntity fileEntity : fileEntities) {
                    i++;
                    // 计算并报告进度百分比
                    int progress = (i * 100) / fileEntities.size();
                    //将进度保存到数据库
                    ProgressListener listener = ProgressContext.getProgressListener();
                    if (listener != null) {
                        listener.onProgress(algorithmTaskEntity,progress);
                    }
                    String downloadPath = accessAddress + dataSonEntity.getFatherId() + "/v" + dataSonEntity.getVersion()+"/source/";
                    algorithmParams.setImage_path(downloadPath+"/"+ fileEntity.getFdName());
                    algorithmParams.setAlgorithmId(algorithmId);
                    algorithmParams.setParams(params);

                    MarkInfoEntity markInfoEntity = markInfoMapper.selectOne(new LambdaQueryWrapper<MarkInfoEntity>().eq(MarkInfoEntity::getSonId, sonId)
                            .eq(MarkInfoEntity::getFileId, fileEntity.getId()));

                    if(ObjectUtil.isNotEmpty(markInfoEntity)){
                        if(ObjectUtil.isNotEmpty(markInfoEntity.getLabelMarkInfo())){
                            algorithmParams.setMarkInfo(markInfoEntity.getLabelMarkInfo());
                            algorithmParams.setImageAbsoute(fileEntity.getFdPath());
                        }
                        fileEntity.setOperateHeight(markInfoEntity.getOperateHeight());
                        fileEntity.setOperateWidth(markInfoEntity.getOperateWidth());
                    }else{
                        if(algorithmEntity.getId()==4){
                            algorithmTaskEntity.setTaskException("此任务需要标注信息,请标注所有文件后重试");
                            algorithmTaskEntity.setTaskProgress("100%");
                            algorithmTaskEntity.setTaskStat("异常");
                            algorithmTaskService.updateById(algorithmTaskEntity);
                            continue;
                        }
                    }

                    String genResult = null;
                    try{
                        genResult = modelUtil.model(algorithmEntity, algorithmParams);
                    }catch (Exception e){
                        genResult = e.getMessage();
                    }



                    AlgorithmTaskResultEntity algorithmTaskResultEntity = new AlgorithmTaskResultEntity();
                    algorithmTaskResultEntity.setTaskId(algorithmTaskEntity.getTaskId());
                    algorithmTaskResultEntity.setTaskParams(JSONUtil.toJsonStr(algorithmParams));
                    algorithmTaskResultEntity.setTaskResult(JSONUtil.isJson(genResult)?JSONUtil.toJsonStr(genResult):genResult);
                    algorithmTaskResultMapper.insert(algorithmTaskResultEntity);
                    //读取genResult
                    String dataBaseParams = algorithmEntity.getResponseParams();
                    //解析params 从数据结中匹配
                    if(ObjectUtil.isEmpty(dataBaseParams) || !JSONUtil.isJson(genResult)){
                        continue;
                    }
                    JSONArray jsonArray = JSONUtil.parseArray(dataBaseParams);
                    List<Map> dataBaseResponseParamsMap = JSONUtil.toList(jsonArray, Map.class);
                    //将数组组装到map
                    Map<String,List> resultMap =  new HashMap<>();
                    for (Map responseMap : dataBaseResponseParamsMap) {
                        String key = responseMap.get("serverKey").toString();
                        if (responseMap.get("type").equals("path")) {

                            String fileDirOrName = "";
                            try{
                                fileDirOrName =  JSONUtil.parseObj(genResult).get(key).toString();
                            }catch (Exception e){
                                continue;
                            }
//                            String prefixToRemove = "/home/taizun";

                            // 使用 replaceFirst 方法去掉前缀
//                                fileDirOrName = URLUtils.encodeURL(algorithmPicHttp+ fileDirOrName.replaceFirst("^" + prefixToRemove, ""));
                            if(algorithmEntity.getId().toString().equalsIgnoreCase("9") || algorithmEntity.getId().toString().equalsIgnoreCase("10")){
                                fileDirOrName = URLUtils.encodeURL(algorithmPicHttp1+ fileDirOrName.replaceFirst("^" + prefixToRemove1, ""));
                            }else{
                                fileDirOrName = URLUtils.encodeURL(algorithmPicHttp+ fileDirOrName.replaceFirst("^" + prefixToRemove, ""));
                            }

                            System.out.println("目录为"+fileDirOrName);
                            List<FileEntity> result = new ArrayList<>();
                            if (isDirectory(fileDirOrName) ) {
                                List<String> fileURLs = getFileURLs(fileDirOrName);
                                if(fileURLs.size()>0){
                                    for (String fileURL : fileURLs) {
                                        List<FileEntity> extracted = extracted(result, fileEntity, taskId, fileURL, newFilePath);
                                        resultMap.put(key,extracted);
                                    }
                                }

                                System.out.println("===================");

                            }else if(isImageFile(fileDirOrName)){
                                List<FileEntity> extracted = extracted(result, fileEntity, taskId, fileDirOrName, newFilePath);
                                resultMap.put(key,extracted);
                            }else{
//                                    List<FileEntity> extracted = extracted(result, fileEntity, taskId, fileDirOrName, newFilePath);
//                                    resultMap.put(key,extracted);
                                System.out.println("=======================================不是目录不是图片");
                            }
                            algorithmTaskEntity.setResult(result);
                            System.out.println("========1111111111==========");
                            dataSonService.updateById(dataSonEntity);
                            System.out.println("=======22222222222==========");

                        }
                    }

                    if(algorithmEntity.getId() == 9 || algorithmEntity.getId() == 10){
                        //获取label json文件id 和 文件 id
                        Map<String,Object> lableAndFileIdMap = new HashMap<>();
                        List<FileEntity> imageList = resultMap.get("original_image");
                        List<FileEntity> labelList = resultMap.get("label_json");
                        List<FileEntity> markList = resultMap.get("output_image");
                        if(ObjectUtil.isNotEmpty(labelList) && labelList.size() > 0){
                            if(labelList.size() == 1){
                                FileEntity fileEntity1 = labelList.get(0);
                                Integer id = fileEntity1.getId();
                                lableAndFileIdMap.put("jsonId",id+"");
                            }else{
                                List<String> ids = new ArrayList<>();
                                for (FileEntity entity : labelList) {
                                    ids.add( entity.getId() + "");
                                }
                                lableAndFileIdMap.put("jsonId",ids);
                            }
                        }
                        if(ObjectUtil.isNotEmpty(markList) && markList.size() > 0){
                            if(markList.size() == 1){
                                FileEntity fileEntity1 = markList.get(0);
                                Integer id = fileEntity1.getId();
                                lableAndFileIdMap.put("markId",id+"");
                            }else{
                                List<String> ids = new ArrayList<>();
                                for (FileEntity entity : markList) {
                                    ids.add( entity.getId() + "");
                                }
                                lableAndFileIdMap.put("markId",ids);
                            }
                        }
//                            String ytId = lableAndFileIdMap.get("ytId").toString();
                        String ytId =fileEntity.getId()+"";
                        String markId = lableAndFileIdMap.get("markId").toString();
                        String jsonId = lableAndFileIdMap.get("jsonId").toString();

                        // 此处调用hh方法 将标注信息入库
                        dataMarkService.setMarkFileJsonAndMarkFileWrite(sonId,ytId,jsonId,markId);

                    }

                }
//                    dataMarkService.setMarkFileJsonWrite(datasetOutId);
                algorithmTaskEntity.setTaskStat("结束");
                algorithmTaskEntity.setTaskProgress("100%");
                myWebSocketHandler.sendMessageToUser(BizConstants.TASK_PROGRESS,taskId+"", JSONUtil.toJsonStr(algorithmTaskEntity));
                // 检查消息内容，决定是否断开连接
                myWebSocketHandler.disconnectUser(BizConstants.TASK_PROGRESS, taskId+"");
                algorithmTaskService.saveOrUpdate(algorithmTaskEntity);

            }catch (Exception e){

                algorithmTaskEntity.setTaskStat("异常");
                algorithmTaskEntity.setTaskException("异常:"+e.getMessage());
                myWebSocketHandler.sendMessageToUser(BizConstants.TASK_PROGRESS,taskId+"",JSONUtil.toJsonStr(algorithmTaskEntity));
                myWebSocketHandler.disconnectUser(BizConstants.TASK_PROGRESS, taskId+"");
                algorithmTaskEntity.setTaskProgress("100%");
                algorithmTaskService.saveOrUpdate(algorithmTaskEntity);
            }
        }

    }


    private void extracted(String taskId, String fileUrlName,String newFilePath) throws IOException {
        log.info("开始保存====");
        String fileAboslute = downloadFile(fileUrlName, newFilePath);
        File file = new File(fileAboslute);
        FileEntity fileEntity1 = new FileEntity();
        fileEntity1.setFdPath(newFilePath+file.getName());
        fileEntity1.setFdName(file.getName());
        fileEntity1.setFileStatus(1);
        fileEntity1.setTaskId(taskId);
        fileEntity1.setFdType(Files.probeContentType(file.toPath()));
        fileEntity1.setFdSize(FileFormatSizeUtils.formatSize(file.length()));
        fileEntity1.setFdSuffix(Objects.requireNonNull(file.getName()).substring(file.getName().lastIndexOf(".")));
        fileEntity1.setFdAccessPath(file.getName());

//        boolean imageFile = isImageFile(fileUrlName);
        boolean save = fileService.save(fileEntity1);
        System.out.println("处理过文件插入结果:"+ save);
//        if(save){
//            String fileIds = dataSon.getFileIds();
//            if (StringUtils.isEmpty(fileIds)){
//                dataSon.setFileIds(fileEntity1.getId()+"");
//            }else{
//                dataSon.setFileIds(fileIds+ ","+fileEntity1.getId());
//            }
//        }
    }
    private List<FileEntity>  extracted(List<FileEntity> resultLst,FileEntity fileEntity,Long taskId, String fileUrlName,String newFilePath) throws IOException {
        String fileAboslute = downloadFile(fileUrlName, newFilePath);
        File file = new File(fileAboslute);
        FileEntity fileEntity1 = new FileEntity();
        fileEntity1.setFdPath(newFilePath+file.getName());
        fileEntity1.setFdName(file.getName());
        fileEntity1.setFileStatus(1);
        fileEntity1.setFdType(Files.probeContentType(file.toPath()));
        fileEntity1.setFdSize(FileFormatSizeUtils.formatSize(file.length()));
        fileEntity1.setFdSuffix(Objects.requireNonNull(file.getName()).substring(file.getName().lastIndexOf(".")));
        fileEntity1.setFdAccessPath(file.getName());
        fileEntity1.setTaskId(taskId+"");
        fileEntity1.setOperateWidth(fileEntity.getOperateWidth());
        fileEntity1.setOperateHeight(fileEntity.getOperateHeight());

//        boolean imageFile = isImageFile(fileUrlName);
        boolean save = fileService.save(fileEntity1);
        System.out.println("处理过文件插入结果:"+ save);
        resultLst.add(fileEntity1);
        return resultLst;
//        if(save){
//            String fileIds = dataSon.getFileIds();
//            if (StringUtils.isEmpty(fileIds)){
//                dataSon.setFileIds(fileEntity1.getId()+"");
//            }else{
//                dataSon.setFileIds(fileIds+ ","+fileEntity1.getId());
//            }
//        }
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
//http://10.5.28.222/outputs/%25E5%25BE%25AE%25E4%25BF%25A1%25E5%259B%25BE%25E7%2589%2587_20240906105012.jpg
// %25E5%25BE%25AE%25E4%25BF%25A1%25E5%259B%25BE%25E7%2589%2587_20240906105017%2520-%2520%25E5%2589%25AF%25E6%259C%25AC%2520-%2520%25E5%2589%25AF%25E6%259C%25AC%2520-%2520%25E5%2589%25AF%25E6%259C%25AC%2520-%2520%25E5%2589%25AF%25E6%259C%25AC%2520-%2520%25E5%2589%25AF%25E6%259C%25AC%2520-%2520%25E5%2589%25AF%25E6%259C%25AC.jpg

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
        String url ="http://192.168.1.4/results/20240927_141812/35KV%E6%A1%91%E7%AA%9D%E7%BA%BF016%23%E5%A1%94-%E4%B8%8A%E7%9B%B8%E5%B0%8F%E5%8F%B7%E4%BE%A7%E7%BB%9D%E7%BC%98%E5%AD%90%E6%B6%82%E5%B1%82%E7%A0%B4%E6%8D%9F-DSC-51320/";
        String s = URLUtils.encodeURL(url);
        System.out.println(s);
        System.out.println(isDirectory(s));
        List<String> fileURLs = getFileURLs(s);
        for (String fileURL : fileURLs) {
            System.out.println(downloadFile(fileURL,"."));
        }
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

    @Async
    @MonitorProgress
    public void execDtaaEnhancementTask(AlgorithmTaskEntity algorithmTaskEntity) {
        //下载数据集
        uploadDataSet(algorithmTaskEntity);


        //增强参数map
        // {"operate":"","model":"1","num_aug":"1"}

        //将数据库的operate 与 algorithmTaskEntity.getParamsLst()的operate 匹配 获取 相应的参数
        Map<String,String> tmpMap = new HashMap<>();
        log.info("开始解析数据============================");
        algorithmTaskEntity.getParamsLst().stream().forEach(
                item -> {
                    StringBuffer sb = new StringBuffer();
                    if (item.containsKey("algorithmId")) {
                        AlgorithmEntity algorithmId= algorithmService.getById(Integer.parseInt(item.get("algorithmId").toString()));
                        String operate = algorithmId.getOperate();
                        tmpMap.put("url",algorithmId.getUrl());
                        tmpMap.put("num_augmentations",ObjectUtil.isEmpty(algorithmId.getNumAugmentations())?"1":algorithmId.getNumAugmentations());
                        sb.append(operate);
                        item.remove("algorithmId");
                    }
                    //将移除后的对象拼接成字符串"resize:[640, 640];horizontal_flip;random_crop:[640, 640];center_crop:[640, 640];to_gray;affine"
                    if(item.keySet().size()>0){
                        sb.append(":[");
                        int i = 0;
                        for (String key : item.keySet()){
                            if(i > 0){
                                sb.append("," + item.get(key));
                            }else{
                                sb.append( item.get(key));
                            }
                            i++;
                        }
                        sb.append("];");
                    }else{
                        sb.append(";");
                    }

                    String op = ObjectUtil.isEmpty(tmpMap.get("operate")) ? "" :tmpMap.get("operate");
                    tmpMap.put("operate",op+sb.toString());
                }
        );
        tmpMap.put("model",algorithmTaskEntity.getDataEnhanceTactics());
        String url = tmpMap.get("url");
        log.info("解析数据完毕============================"+JSONUtil.toJsonStr(tmpMap));
        tmpMap.remove("url");
        String enhanceResult = httpUtil.post(url,tmpMap);
        log.info("增强数据完毕"+enhanceResult);
        //获取数据集
        LambdaQueryWrapper<DataSonEntity> eq = new LambdaQueryWrapper<DataSonEntity>().eq(
                DataSonEntity::getSonId, algorithmTaskEntity.getDataSetId()
        );
        DataSonEntity dataSonEntity = dataSonService.getOne(eq);

        String newFilePath = formalPath + dataSonEntity.getFatherId() + "/" + "v" + dataSonEntity.getVersion() + "/" + "generate" +  "/" + "图像算子增强"+"/" ;
        File fileMkdir = new File(newFilePath);
        // 判断目录是否存在，不存在，则创建
        if (!fileMkdir.exists()) {
            fileMkdir.mkdirs();
        }
        try {
            saveEnhancementResult(algorithmTaskEntity.getTaskId()+"",enhanceResult,newFilePath);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        algorithmTaskEntity.setTaskStat("结束");
        algorithmTaskEntity.setTaskProgress("100%");
        myWebSocketHandler.sendMessageToUser(BizConstants.TASK_PROGRESS,algorithmTaskEntity.getTaskId()+"", JSONUtil.toJsonStr(algorithmTaskEntity));
        // 检查消息内容，决定是否断开连接
        myWebSocketHandler.disconnectUser(BizConstants.TASK_PROGRESS, algorithmTaskEntity.getTaskId()+"");
        algorithmTaskService.saveOrUpdate(algorithmTaskEntity);



    }

    public void uploadDataSet(AlgorithmTaskEntity algorithmTaskEntity){
        log.info("开始下载文件================");
        String dataSetId = algorithmTaskEntity.getDataSetId();
        DataSonEntity one = dataSonService.getOne(new LambdaQueryWrapper<DataSonEntity>().eq(
                DataSonEntity::getSonId, dataSetId
        ));
        AlgorithmEntity algorithmEntity = algorithmMapper.selectById(29);

        //拿到数据集的目录
        String dataSetPath = formalPath +  one.getFatherId() + "/v" + one.getVersion();
        String dataSetHttpPath = accessAddress  +  one.getFatherId() + "/v" + one.getVersion();
        log.info("数据集目录"+ dataSetPath);
        log.info("数据集http目录"+ dataSetHttpPath);
        // 获取json文件夹中的所有文件
        File jsonFolder = new File(dataSetPath + "/json/");
        File[] jsonFilesArray = null;
        if (jsonFolder.exists() && jsonFolder.isDirectory()) {
            jsonFilesArray = jsonFolder.listFiles();
        }else {
            throw new RuntimeException("该数据集没有json信息，请检查是否标注");
        }

        // 获取source文件夹中的所有文件
        log.info("获取source文件夹中的所有文件================");
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
                                        log.info("调用下载接口================"+algorithmParams);
                                        model = modelUtil.model(algorithmEntity, algorithmParams);
                                        log.info("下载结果"+model);
                                    }catch (Exception e){
                                        recordTaskParams(algorithmTaskEntity,e.getMessage());
                                        throw new RuntimeException("算子下载数据集接口无法调用");

                                    }
                                    recordTaskParams(algorithmTaskEntity,model);
                                }
                            }
                        }
                    }
                }
            }
        }else{
            log.info("目录不存在================");
        }
    }

    public void recordTaskParams( AlgorithmTaskEntity algorithmTaskEntity,String genResult){
        AlgorithmTaskResultEntity algorithmTaskResultEntity = new AlgorithmTaskResultEntity();
        algorithmTaskResultEntity.setTaskId(algorithmTaskEntity.getTaskId());
        algorithmTaskResultEntity.setTaskParams(JSONUtil.toJsonStr(algorithmTaskEntity.getParams()));
        algorithmTaskResultEntity.setTaskResult(JSONUtil.isJson(genResult)?JSONUtil.toJsonStr(genResult):genResult);
        algorithmTaskResultMapper.insert(algorithmTaskResultEntity);
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

    public void saveEnhancementResult(String tskId,String result,String newFilePath) throws IOException {

        String fileDirOrName = "";
        try{
            fileDirOrName =  JSONUtil.parseObj(result).get("path").toString();
        }catch (Exception e){
            throw new RuntimeException("增强结果出错",e);
        }
//                            String prefixToRemove = "/home/taizun";

        // 使用 replaceFirst 方法去掉前缀
//                                fileDirOrName = URLUtils.encodeURL(algorithmPicHttp+ fileDirOrName.replaceFirst("^" + prefixToRemove, ""));
        fileDirOrName = URLUtils.encodeURL(algorithmPicHttp+ fileDirOrName.replaceFirst("^" + prefixToRemove, ""));
        if (isDirectory(fileDirOrName) ) {
            List<String> fileURLs = getFileURLs(fileDirOrName);
            if(fileURLs.size()>0){
                for (String fileURL : fileURLs) {
                    log.info("文件路径"+fileURL);
                    extracted(tskId, fileURL, newFilePath);
                }
            }
            System.out.println("==================="+fileDirOrName);

        }else if(isImageFile(fileDirOrName)){
            extracted(tskId,fileDirOrName, newFilePath);
        }else{
            extracted(tskId,fileDirOrName, newFilePath);
        }
        log.info("增强文件保存成功");
    }
}
