package com.qczy.federated.controller;

import com.qczy.common.result.Result;
import com.qczy.federated.model.FederatedNode;
import com.qczy.federated.model.ModelType;
import com.qczy.federated.model.TrainingJob;
import com.qczy.federated.service.FederatedCoordinatorService;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.Collection;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/federated")
@Api(tags = "联邦学习管理")
public class FederatedController {

    @Autowired
    private FederatedCoordinatorService coordinatorService;

    @PostMapping("/register")
    @ApiOperation("节点注册")
    public Result register(@RequestBody FederatedNode node) {
        return Result.ok(coordinatorService.registerNode(node));
    }

    @PostMapping("/heartbeat/{nodeId}")
    @ApiOperation("节点心跳")
    public Result heartbeat(@PathVariable String nodeId, @RequestBody Map<String, Object> metadata) {
        coordinatorService.heartbeat(nodeId, metadata);
        return Result.ok(true);
    }

    @PostMapping("/jobs")
    @ApiOperation("创建训练任务")
    public Result createJob(@RequestParam ModelType modelType,
                            @RequestBody Map<String, Object> body) {
        Map<String, Object> hyper = (Map<String, Object>) body.get("hyperParameters");
        List<String> nodes = (List<String>) body.get("participantNodeIds");
        Object baseline = body.get("baselineAccuracy");
        Object allowedDrop = body.get("allowedDropPercent");
        TrainingJob job = coordinatorService.createJob(modelType, hyper, nodes);
        if (baseline instanceof Number) {
            job.setBaselineAccuracy(((Number) baseline).doubleValue());
        }
        if (allowedDrop instanceof Number) {
            job.setAllowedDropPercent(((Number) allowedDrop).doubleValue());
        }
        return Result.ok(job);
    }

    @PostMapping("/jobs/{jobId}/start")
    @ApiOperation("启动训练任务")
    public Result start(@PathVariable String jobId) {
        coordinatorService.startJob(jobId);
        return Result.ok(true);
    }

    @PostMapping("/jobs/{jobId}/stop")
    @ApiOperation("停止训练任务")
    public Result stop(@PathVariable String jobId) {
        coordinatorService.stopJob(jobId);
        return Result.ok(true);
    }

    @GetMapping("/nodes")
    @ApiOperation("节点列表")
    public Result nodes() {
        Collection<FederatedNode> list = coordinatorService.listNodes();
        return Result.ok(list);
    }

    @GetMapping("/jobs")
    @ApiOperation("任务列表")
    public Result jobs() {
        Collection<TrainingJob> list = coordinatorService.listJobs();
        return Result.ok(list);
    }
}





