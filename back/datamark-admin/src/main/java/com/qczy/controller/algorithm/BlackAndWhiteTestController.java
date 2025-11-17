package com.qczy.controller.algorithm;

import com.qczy.common.result.Result;
import com.qczy.model.entity.AlgorithmTaskEntity;
import com.qczy.model.request.BWTestRequest;
import com.qczy.model.request.TrainEntity;
import com.qczy.service.BlackAndWhiteTestService;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/algorithm/blackAndWhiteTest")
@Api(tags = "算法==黑白盒测试")
public class BlackAndWhiteTestController {

    @Autowired
    private BlackAndWhiteTestService blackAndWhiteTestService;


    @PostMapping("/start")
    @ApiOperation("黑白盒测试")
    public Result start(BWTestRequest bwTestRequest) {
        try{
            blackAndWhiteTestService.startTest(bwTestRequest);
        }catch (Exception e){
            return Result.fail(e.getMessage());
        }
        return Result.ok("黑白盒测试任务已开始");
    }

    @PostMapping("/result")
    @ApiOperation("黑白盒测试结果")
    public Result result(@RequestBody BWTestRequest bwTestRequest) {

        try{
            return Result.ok(blackAndWhiteTestService.searchResult(bwTestRequest));
        }catch (Exception e){
            return Result.fail(e.getMessage());
        }
    }

    @PostMapping("/getFiles")
    public Result getFiles(){
        return Result.ok();
    }


    @Value(("${upload.formalPath}"))
    private String formalPath;
    @Value(("${file.accessAddress}"))
    private String httpPath;


    /**
     * 处理单个文件上传
     */
    public Result<Map<String, Object>> uploadFile(
            MultipartFile file) {
        Map<String, Object> result = new HashMap<>();

        try {
            // 验证文件是否为空
            if (file.isEmpty()) {
                result.put("success", false);
                result.put("message", "上传文件不能为空");
                return Result.fail(result);
            }

            // 创建上传目录（如果不存在）
            Path uploadDir = Paths.get(formalPath);
            if (!Files.exists(uploadDir)) {
                Files.createDirectories(uploadDir);
            }

            // 生成唯一文件名，避免冲突
            String originalFilename = file.getOriginalFilename();
            String fileExtension = getFileExtension(originalFilename);
            String uniqueFilename = generateUniqueFilename(fileExtension);
            Path filePath = uploadDir.resolve(uniqueFilename);

            // 保存文件
            Files.copy(file.getInputStream(), filePath, StandardCopyOption.REPLACE_EXISTING);

            // 返回成功信息
            result.put("success", true);
            result.put("message", "文件上传成功");
            result.put("fileName", uniqueFilename);
            result.put("fileSize", file.getSize());
            result.put("savePath", filePath.toString());
            result.put("httpPath", httpPath+uniqueFilename);

            return Result.ok(result);

        } catch (IOException e) {
            e.printStackTrace();
            result.put("success", false);
            result.put("message", "文件上传失败: " + e.getMessage());
            return Result.fail(result);
        }
    }

    /**
     * 生成唯一文件名
     */
    private String generateUniqueFilename(String extension) {
        LocalDateTime now = LocalDateTime.now();
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyyMMddHHmmssSSS");
        String timestamp = now.format(formatter);
        return "upload_" + timestamp + extension;
    }

    /**
     * 获取文件扩展名
     */
    private String getFileExtension(String fileName) {
        if (fileName == null || fileName.lastIndexOf(".") == -1) {
            return "";
        }
        return fileName.substring(fileName.lastIndexOf("."));
    }

}