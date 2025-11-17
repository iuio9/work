package com.qczy.service.impl;

import cn.hutool.json.JSONUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.JsonPrimitive;
import com.qczy.common.markInfo.AnalysisMarkInfoUtils;
import com.qczy.mapper.*;
import com.qczy.model.entity.*;
import com.qczy.model.request.DataSonEntityRequest;
import com.qczy.service.FileMarkService;
import com.qczy.utils.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;
import org.springframework.util.CollectionUtils;
import org.springframework.util.ObjectUtils;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/12/25 15:38
 * @Description:
 */
@Service
public class FileMarkServiceImpl implements FileMarkService {

    private static final Logger log = LoggerFactory.getLogger(FileMarkServiceImpl.class);

    @Autowired
    private DataSonMapper dataSonMapper;

    @Autowired
    private FileMapper fileMapper;

    @Autowired
    private MarkInfoMapper markInfoMapper;

    @Autowired
    private DataImportLogMapper dataImportLogMapper;

    @Autowired
    private DataSonLabelMapper dataSonLabelMapper;

    @Autowired
    private LabelMapper labelMapper;

    @Autowired
    private MyHaoWebSocketUtils myHaoWebSocketUtils;

    @Autowired
    private FileDownloadUtils fileDownloadUtils;

    @Value("${upload.formalPath}")
    private String formalPath;


    @Override
    public void addMarkSon(DataSonEntityRequest dataSonEntityRequest, String fileIdsStr, Date startDate, int sumCount, int currentCount) {
        DataSonEntity dataSonEntity = dataSonMapper.selectOne(
                new LambdaQueryWrapper<DataSonEntity>().eq(DataSonEntity::getSonId, dataSonEntityRequest.getSonId())
        );
        if (ObjectUtils.isEmpty(dataSonEntity)) {
            return;
        }
        String[] fileIds = fileIdsStr.split(",");
        Map<String, Integer> jsonMap = new HashMap<>();  //json文件集合  (key：文件名 、 value：文件id )
        Map<String, Integer> xmlMap = new HashMap<>();   //xml文件集合  (key：文件名 、 value：文件id )
        Map<String, Integer> imgMap = new HashMap<>();   //图片文件集合   (key：文件名 、 value：文件id )
        StringBuilder sb = new StringBuilder();          //记录新增成功的文件的id
        int fileSumCount = 0;
        int fileCurrentCount = 0;
        // 临时变量
        int j = 0;

        Date endDate = null;
        try {
            for (String fileId : fileIds) {
                // 判断文件后缀
                FileEntity fileEntity = fileMapper.selectById(Integer.parseInt(fileId));
                if (ObjectUtils.isEmpty(fileEntity)) {
                    continue;
                }

                // 存入对应的集合里
                if (fileEntity.getFdName().endsWith("json")) {
                    jsonMap.put(fileEntity.getFdName(), fileEntity.getId());
                    continue;
                }

                if (fileEntity.getFdName().endsWith("xml")) {
                    xmlMap.put(fileEntity.getFdName(), fileEntity.getId());
                    continue;
                }

                if (ImageUtils.isImage(fileEntity.getFdPath())) {
                    imgMap.put(fileEntity.getFdName(), fileEntity.getId());
                }
            }


            fileSumCount = imgMap.size() * 2; // 这个值为进度条使用的
            fileCurrentCount = imgMap.size();


            if (!jsonMap.isEmpty()) {
                System.out.println("-------------------------------" + jsonMap);
                Iterator<Map.Entry<String, Integer>> iterator = jsonMap.entrySet().iterator();
                while (iterator.hasNext()) {
                    Map.Entry<String, Integer> entity = iterator.next();
                    FileEntity fileEntity = fileMapper.selectById(entity.getValue());

                    Map<String, List<WebRectangleShape>> map = null;
                    // TODO 判断走哪个文件
                    if (returnStatus(fileEntity.getFdPath()) == 1) {
                        map = AnalysisMarkInfoUtils.analysisJson3(fileEntity.getFdPath());
                    } else {
                        map = AnalysisMarkInfoUtils.analysisJson4(fileEntity.getFdPath(), imgMap);
                    }
                    if (map == null || map.isEmpty()) {
                        continue;
                    }
                    for (Map.Entry<String, List<WebRectangleShape>> markEntityMap : map.entrySet()) {
                        System.out.println("---------------------图片名称：" + markEntityMap.getKey() + "---------------------------");
                        System.out.println(markEntityMap);
                        if (imgMap.containsKey(markEntityMap.getKey())) {
                            Integer imgId = imgMap.get(markEntityMap.getKey());
                            List<WebRectangleShape> rectangleShapes = markEntityMap.getValue();
                            if (imgId != null && rectangleShapes != null) {
                                fileCurrentCount = saveMarkSon(dataSonEntity, imgId, rectangleShapes, sb, fileSumCount, fileCurrentCount);
                                imgMap.remove(markEntityMap.getKey());
                            }
                        }
                    }
                }
                // 计算当前文件总数量
                sumCount = sumCount - (jsonMap.size());
                // 使用迭代器的 remove 方法删除元素
                iterator.remove();

            }


            if (!xmlMap.isEmpty()) {
                System.out.println("-------------------------------" + xmlMap);
                Iterator<Map.Entry<String, Integer>> iterator = xmlMap.entrySet().iterator();
                while (iterator.hasNext()) {
                    Map.Entry<String, Integer> entity = iterator.next();
                    FileEntity fileEntity = fileMapper.selectById(entity.getValue());
                    Map<String, List<WebRectangleShape>> map = AnalysisMarkInfoUtils.analysisXml(fileEntity.getFdPath(), imgMap);
                    if (map == null || map.isEmpty()) {
                        continue;
                    }
                    for (Map.Entry<String, List<WebRectangleShape>> markEntityMap : map.entrySet()) {
                        System.out.println("------------------------------------------------");
                        System.out.println(markEntityMap);
                        if (imgMap.containsKey(markEntityMap.getKey())) {
                            Integer imgId = imgMap.get(markEntityMap.getKey());
                            List<WebRectangleShape> rectangleShapes = markEntityMap.getValue();
                            if (imgId != null && rectangleShapes != null) {
                                fileCurrentCount = saveMarkSon(dataSonEntity, imgId, rectangleShapes, sb, fileSumCount, fileCurrentCount);
                                imgMap.remove(markEntityMap.getKey());
                            }
                        }
                    }
                }
                // 计算当前文件总数量
                sumCount = sumCount - (xmlMap.size());
                // 使用迭代器的 remove 方法删除元素
                iterator.remove();
            }

            if (!imgMap.isEmpty()) {
                System.out.println("----------------------------------" + imgMap);
                Iterator<Map.Entry<String, Integer>> iterator = imgMap.entrySet().iterator();
                while (iterator.hasNext()) {
                    Map.Entry<String, Integer> entity = iterator.next();
                    FileEntity fileEntity = fileMapper.selectById(entity.getValue());
                    sb.append(fileEntity.getId()).append(",");
                    j++;
                    // 删除元素
                    iterator.remove();


                    // 发送进度条， 使用webSocket 推送实时进度
                    fileCurrentCount++;
                    myHaoWebSocketUtils.sendMessage(
                            dataSonEntityRequest.getSonId()
                            , fileSumCount,  //总数量
                            fileCurrentCount); // 当前数量
                    System.out.println("sumCount=" + fileSumCount + " currentCount=" + fileCurrentCount);
                    System.out.println("数据集id-----------------------" + dataSonEntityRequest.getSonId());
                    System.out.println("发送进度条----------------------sumCount" + fileSumCount + ",currentCount=" + fileCurrentCount);

                }
            }

            endDate = new Date();
            System.out.println("sumCount:" + sumCount);
            System.out.println("sb:" + sb);

            // 首先先发送进度 //100%
            myHaoWebSocketUtils.sendMessage(dataSonEntityRequest.getSonId(), 100);
            savaDataImportLog(dataSonEntityRequest, startDate, endDate, sb, sumCount, j, imgMap, jsonMap, xmlMap);
        } catch (Exception e) {


            myHaoWebSocketUtils.sendMessage(dataSonEntityRequest.getSonId(), 100);
            DataImportLogEntity importLogEntity = new DataImportLogEntity();
            importLogEntity.setSonId(dataSonEntityRequest.getSonId());
            importLogEntity.setFileSize("0");
            importLogEntity.setFileIds("");
            importLogEntity.setStatus(2);
            importLogEntity.setImportStartTime(startDate);
            importLogEntity.setImportEndTime(new Date());
            importLogEntity.setCreateTime(new Date());
            importLogEntity.setUserId(dataSonEntityRequest.getUserId());
            int result = dataImportLogMapper.insert(importLogEntity);

            log.error("错误日志：" ,e.getMessage());
            e.printStackTrace();
            System.out.println(result > 0 ? "日志新增成功！................" : "日志新增失败！.....................");


            if (StringUtils.isEmpty(dataSonEntityRequest.getOldFileIds())) {
                dataSonEntity.setFileIds("");
                dataSonEntity.setStatus("0% (0/0)");
            } else {
                dataSonEntity.setFileIds(dataSonEntityRequest.getOldFileIds());
            }
            dataSonMapper.updateById(dataSonEntity);
            //throw new RuntimeException(e);

        }/* finally {
            // TODO 无论是否异常，必须处理的
            // 首先先发送进度 //100%
            myHaoWebSocketUtils.sendMessage(dataSonEntityRequest.getSonId(), 100);

        }*/

    }


    // 新增方法
    private Integer saveMarkSon(DataSonEntity dataSonEntity, Integer fileId, List<WebRectangleShape> markInfo, StringBuilder sb, int fileSumCount, int fileCurrentCount) {
        String sonId = dataSonEntity.getSonId();
        sb.append(fileId).append(",");
        MarkInfoEntity markInfoEntity = new MarkInfoEntity();
        markInfoEntity.setSonId(sonId);
        markInfoEntity.setFileId(fileId);
        markInfoEntity.setMarkFileId(fileId);
        FileEntity fileEntity = fileMapper.selectById(fileId);
        markInfoEntity.setWidth(fileEntity.getWidth());
        markInfoEntity.setHeight(fileEntity.getHeight());
        markInfoEntity.setOperateWidth(fileEntity.getWidth());
        markInfoEntity.setOperateHeight(fileEntity.getHeight());
        markInfoEntity.setMarkInfo(JSONUtil.toJsonStr(markInfo));

        if (!StringUtils.isEmpty(markInfoEntity.getMarkInfo())) {
            List<WebRectangleShape> list = JSONUtil.toList(JSONUtil.parseArray(markInfo), WebRectangleShape.class);
            List<WebRectangleShape> point = list.stream().filter(item -> {
                return item.getType().equalsIgnoreCase("point");
            }).collect(Collectors.toList());
            List<WebRectangleShape> mark = list.stream().filter(item -> {
                return !item.getType().equalsIgnoreCase("point");
            }).collect(Collectors.toList());
            LabelmeImageData labelmeImageData = FormatConverter.convertWebDataToLabelme(mark, fileEntity.getFdName(), fileEntity.getWidth(),
                    fileEntity.getHeight(), markInfoEntity);
            markInfoEntity.setLabelMarkInfo(JSONUtil.toJsonStr(labelmeImageData));
        }


        markInfoEntity.setCreateTime(new Date());
        // 解析标签
        String labelIds = setDataSetLabel(markInfo, sonId);
        if (!StringUtils.isEmpty(labelIds)) {
            markInfoEntity.setLabels(labelIds);
        }

        markInfoMapper.insert(markInfoEntity);

        // 生成json文件
        generateFile(dataSonEntity, markInfoEntity, fileEntity);


        // 发送进度条， 使用webSocket 推送实时进度
        fileCurrentCount++;

        myHaoWebSocketUtils.sendMessage(
                sonId
                , fileSumCount,  //总数量
                fileCurrentCount); // 当前数量
        System.out.println("sumCount=" + fileSumCount + " currentCount=" + fileCurrentCount);
        System.out.println("数据集id-----------------------" + sonId);
        System.out.println("发送进度条----------------------sumCount" + fileSumCount + ",currentCount=" + fileCurrentCount);

        return fileCurrentCount;
    }

    // 记录数据集导入日志
    private void savaDataImportLog(DataSonEntityRequest dataSonEntityRequest, Date startData, Date endData, StringBuilder sb, int sumCount, int j
            , Map<String, Integer> imgMap, Map<String, Integer> jsonMap, Map<String, Integer> xmlMap) {
        //TODO ----------------------------------------------- ①计算文件id的数量 -----------------------------------------------

        String fileIds = "";
        if (!StringUtils.isEmpty(sb)) {
            fileIds = sb.deleteCharAt(sb.length() - 1).toString();
        }

        //TODO  判断是否解析成功
        StringBuilder sbFileIds = new StringBuilder(fileIds);
        Collection<Integer> values = imgMap.values();
        String noFileIds = values.stream()
                .map(Object::toString)
                .collect(Collectors.joining(","));
        if (!StringUtils.isEmpty(noFileIds)) {
            if (sbFileIds.length() > 0) {
                sbFileIds.append(",");
            }
            sbFileIds.append(noFileIds);
        }
        fileIds = sbFileIds.toString();

        // 修改数据集文件id
        DataSonEntity dataSonEntity = dataSonMapper.selectById(dataSonEntityRequest.getId());
        System.out.println(dataSonEntity);

        //TODO ----------------------------------------------- ②更改数据集的文件id，标注进度 -----------------------------------------------

        // 文件总数量
        // int imgNum = sumCount - jsonMap.size() - xmlMap.size();
        int imgNum = sumCount;
        // 标注信息上传成功的总数量 //TODO 如果之前数据集已经有图片，且是标注过的，也计算在内
        int markCount = markInfoMapper.selectCount(
                new LambdaQueryWrapper<MarkInfoEntity>()
                        .eq(MarkInfoEntity::getSonId, dataSonEntity.getSonId())
        ) == null ? 0 : markInfoMapper.selectCount(
                new LambdaQueryWrapper<MarkInfoEntity>()
                        .eq(MarkInfoEntity::getSonId, dataSonEntity.getSonId())
        );

        if (StringUtils.isEmpty(dataSonEntityRequest.getOldFileIds())) {
            dataSonEntity.setFileIds(fileIds);
            int progress = MyProgressUtils.calculateCount(markCount, imgNum);
            dataSonEntity.setStatus(progress + "%" + "(" + markCount + "/" + imgNum + ")");

        } else { // 之前有过上传的
            imgNum += dataSonEntityRequest.getOldFileIds().split(",").length;
            fileIds = dataSonEntityRequest.getOldFileIds() + "," + fileIds;
            dataSonEntity.setFileIds(fileIds);
            int progress = MyProgressUtils.calculateCount(markCount, imgNum);
            dataSonEntity.setStatus(progress + "%" + "(" + markCount + "/" + imgNum + ")");
        }
        dataSonMapper.updateById(dataSonEntity);


        //TODO ----------------------------------------------- ③记录导入日志 -----------------------------------------------
        DataImportLogEntity importLogEntity = new DataImportLogEntity();
        importLogEntity.setFileSize(FileFormatSizeUtils.formatSize(0));
        importLogEntity.setStatus(1);  // 导入成功
        importLogEntity.setImportStartTime(startData);
        importLogEntity.setImportEndTime(endData);
        importLogEntity.setSonId(dataSonEntityRequest.getSonId());
        importLogEntity.setCreateTime(new Date());
        importLogEntity.setUserId(dataSonEntityRequest.getUserId());
        importLogEntity.setFileIds(fileIds);
        dataImportLogMapper.insert(importLogEntity);

        // TODO -------------------------------- ④ 执行删除 源文件 （.xml 、 .json） ------------------------------------
        String source = formalPath + dataSonEntity.getFatherId() + "/" + "v" + dataSonEntity.getVersion() + "/" + "source";
        deleteFilesWithExtensions(source);

    }

    @Autowired
    private DataMarkServiceImpl dataMarkServiceImpl;

    // 生成文件方法
    public void generateFile(DataSonEntity dataSonEntity, MarkInfoEntity entity, FileEntity fileEntity) {
        dataMarkServiceImpl.processMarkInfo(dataSonEntity, entity, fileEntity, entity.getMarkInfo(), entity.getWidth(), entity.getHeight());
    }


    // 绑定 数据集 的 标签
    private String setDataSetLabel(List<WebRectangleShape> webRectangleShapes, String sonId) {
        if (CollectionUtils.isEmpty(webRectangleShapes)) {
            return "";
        }
        StringBuilder labelIds = new StringBuilder();
        for (WebRectangleShape webRectangleShape : webRectangleShapes) {
            if (ObjectUtils.isEmpty(webRectangleShape.getProps())) {
                continue;
            }
            if (StringUtils.isEmpty(webRectangleShape.getProps().getName())) {
                continue;
            }

            // 查询是否存在
            if (dataSonLabelMapper.selectBySonIdAndLabelNameCount(sonId, webRectangleShape.getProps().getName()) > 0) {
                continue;
            }

            // 实现新增
            LabelEntity labelEntity = new LabelEntity();
            labelEntity.setLabelName(webRectangleShape.getProps().getName());
            labelEntity.setEnglishLabelName(webRectangleShape.getProps().getName());
            labelEntity.setLabelColor("#7C0DDD");
            labelEntity.setLabelGroupId(0);
            labelEntity.setCreateTime(new Date());
            labelEntity.setLabelGroupId(0);
            labelMapper.insert(labelEntity);

            // 关联数据集 AND 标签表
            DataSonLabelEntity dataSonLabelEntity = new DataSonLabelEntity();
            dataSonLabelEntity.setSonId(sonId);
            dataSonLabelEntity.setLabelId(labelEntity.getId());
            dataSonLabelMapper.insert(dataSonLabelEntity);
            labelIds.append(labelEntity.getId()).append(",");
        }
        if (!StringUtils.isEmpty(labelIds.toString())) {
            labelIds.deleteCharAt(labelIds.length() - 1);
        }
        return labelIds.toString();
    }

    // 解析失败，则直接去更新上传好的文件信息
    private void errorUpdateSon(DataSonEntityRequest dataSonEntityRequest, Map<String, Integer> imgMap) {
        // 先获取数据集
        DataSonEntity dataSonEntity = dataSonMapper.selectById(dataSonEntityRequest.getId());
        System.out.println(imgMap.values());
        Collection<Integer> values = imgMap.values();
        String fileIds = values.stream()
                .map(Object::toString)
                .collect(Collectors.joining(","));

        dataSonEntity.setFileIds(fileIds);
        int result = dataSonMapper.updateById(dataSonEntity);
        System.out.println("------------------------------------修改结果：" + result);
    }


    // 删除数据集下的文件
    @Async
    public void deleteFilesWithExtensions(String folderPath) {
        String[] extensions = {".json", ".xml"};
        File folder = new File(folderPath);
        if (folder.exists() && folder.isDirectory()) {
            File[] files = folder.listFiles();
            if (files != null) {
                for (File file : files) {
                    if (file.isFile()) {
                        String fileName = file.getName();
                        for (String ext : extensions) {
                            if (fileName.endsWith(ext)) {
                                file.delete();
                                break;
                            }
                        }
                    }
                }
            }
        }
    }


    public int returnStatus(String filePath) {
        File file = new File(filePath);
        if (!file.exists()) {
            throw new IllegalArgumentException("文件不存在：" + filePath);
        }

        if (!isJsonFile(file)) {
            throw new IllegalArgumentException("文件类型错误，必须为JSON：" + filePath);
        }

        try {
            String jsonContent = new String(Files.readAllBytes(Paths.get(filePath)));
            JsonObject jsonObject = JsonParser.parseString(jsonContent).getAsJsonObject();

            // 检查 imageWidth 和 imageHeight 是否存在且为有效数值
            boolean hasWidth = isValidNumber(jsonObject, "imageWidth");
            boolean hasHeight = isValidNumber(jsonObject, "imageHeight");

            // 逻辑：两个字段都不存在返回0，否则返回1
            return hasWidth || hasHeight ? 1 : 2;

        } catch (IOException e) {
            throw new RuntimeException("读取文件失败：" + filePath, e);
        }
    }

    /**
     * 检查字段是否存在且为有效数值（非null、非JsonNull、数值类型）
     */
    private boolean isValidNumber(JsonObject jsonObject, String fieldName) {
        JsonElement element = jsonObject.get(fieldName);
        return element != null && !element.isJsonNull() && element.isJsonPrimitive() &&
                ((JsonPrimitive) element).isNumber();
    }

    /**
     * 检查文件是否为JSON格式
     */
    private boolean isJsonFile(File file) {
        String fileName = file.getName();
        int dotIndex = fileName.lastIndexOf('.');
        return dotIndex > 0 && "json".equalsIgnoreCase(fileName.substring(dotIndex + 1));
    }


}


