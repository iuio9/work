package com.qczy.controller.model;

import cn.hutool.core.util.ObjectUtil;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.common.result.Result;
import com.qczy.model.entity.AlgorithmEntity;
import com.qczy.service.AlgorithmService;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

/**
 * @Author: gwj
 * @Version: 1.0
 * @Date: 2024/8/22 20:01
 * @Description:
 */
@RestController
@RequestMapping("/algorithm/example")
@Api(tags = "算法实例管理")
public class AlgorithmController {

    @Autowired
    private AlgorithmService algorithmService;

    /**
     * 算法实例列表
     */
    @PostMapping("/getExampleList")
    @ApiOperation("算法实例列表")
    public Result getlExampleList(
                                     @RequestBody AlgorithmEntity modelEntity) {
        return Result.ok(algorithmService.getExampleList(modelEntity));
    }


    @PostMapping("/getExamplePage")
    @ApiOperation("算法实例分页")
    public Result getPage(@RequestParam(value = "page", defaultValue = "1") int page,
                                     @RequestParam(value = "limit", defaultValue = "10") int limit,
                                     @ModelAttribute AlgorithmEntity modelEntity) {
        Page<AlgorithmEntity> pageParam = new Page<>(page, limit);

        return Result.ok(algorithmService.getExamplePage(pageParam,modelEntity));
    }



    /**
     *  查询算法实例
     */
    @PostMapping("/getExampleDetail")
    @ApiOperation("算法实例详情")
    public Result getExampleDetail(@RequestBody AlgorithmEntity modelEntity) {
        if (ObjectUtil.isEmpty(modelEntity)) {
            return Result.fail("校验参数！");
        }
        AlgorithmEntity exampleDetails = algorithmService.getExampleDetails(modelEntity);

        return Result.ok();
    }

    /**
     *  保存标注信息
     */
    @PostMapping("/addExample")
    @ApiOperation("添加算法记录")
    public Result addExample(@RequestBody AlgorithmEntity modelEntity){
        return Result.ok(algorithmService.addExampleInfo(modelEntity));
    }


    /**
     *  保存标注信息
     */
    @PostMapping("/updateExample")
    @ApiOperation("修改算法记录")
    public Result updateExample(@RequestBody AlgorithmEntity modelEntity){
        return Result.ok(algorithmService.editExampleInfo(modelEntity));
    }

    /**
     *  保存标注信息
     */
    @PostMapping("/delExample")
    @ApiOperation("删除算法实例记录")
    public Result delExampleInfo(@RequestBody AlgorithmEntity modelEntity){
        if(ObjectUtil.isEmpty(modelEntity.getId())){
            return Result.fail("算法编号不能为空");
        }
        return Result.ok(algorithmService.delExampleInfo(modelEntity));
    }



}
