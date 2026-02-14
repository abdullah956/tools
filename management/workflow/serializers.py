"""Serializers for the workflow app."""

import uuid

from rest_framework import serializers

from .models import Edge, Node, Workflow


class WorkflowRequestSerializer(serializers.Serializer):
    """Serializer for handling workflow requests."""

    query = serializers.CharField()
    tools = serializers.ListField(child=serializers.CharField(), allow_empty=False)


class NodeSerializer(serializers.ModelSerializer):
    """Serializer for the Node model."""

    class Meta:
        """Meta class for NodeSerializer."""

        model = Node
        fields = "__all__"  # or specify the fields you want to include


class EdgeSerializer(serializers.ModelSerializer):
    """Serializer for the Edge model."""

    class Meta:
        """Meta class for EdgeSerializer."""

        model = Edge
        fields = "__all__"  # or specify the fields you want to include


class WorkflowSerializer(serializers.ModelSerializer):
    """Serializer for the Workflow model."""

    owner_name = serializers.SerializerMethodField()
    owner_unique_id = serializers.SerializerMethodField()

    class Meta:
        """Meta class for WorkflowSerializer."""

        model = Workflow
        fields = [
            "id",
            "name",
            "description",
            "owner",
            "user_query",
            "owner_name",
            "owner_unique_id",
            "created_at",
            "updated_at",
        ]  # Include 'author' if needed
        read_only_fields = [
            "owner",
            "owner_name",
            "owner_unique_id",
            "created_at",
        ]  # Make 'owner' read-only

    def get_owner_name(self, obj):
        """Return the username of the workflow owner."""
        return obj.owner.username

    def get_owner_unique_id(self, obj):
        """Return the unique_id of the workflow owner."""
        return str(obj.owner.unique_id)

    def validate_author(self, value):
        """Validate that the author is a string."""
        if not isinstance(value, str):
            raise serializers.ValidationError("Author must be of type str")
        return value


class WorkflowSaveSerializer(serializers.ModelSerializer):
    """Serializer for saving a workflow with nodes and edges."""

    nodes = NodeSerializer(many=True)
    edges = EdgeSerializer(many=True)

    class Meta:
        """Meta class for WorkflowSaveSerializer."""

        model = Workflow
        fields = [
            "id",
            "name",
            "description",
            "owner",
            "user_query",
            "nodes",
            "edges",
        ]
        read_only_fields = ["owner"]

    def create(self, validated_data):
        """Create a workflow with associated nodes and edges."""
        nodes_data = validated_data.pop("nodes")
        edges_data = validated_data.pop("edges")
        workflow = Workflow.objects.create(**validated_data)

        for node_data in nodes_data:
            Node.objects.create(workflow=workflow, **node_data)

        for edge_data in edges_data:
            Edge.objects.create(workflow=workflow, **edge_data)

        return workflow


class WorkflowSaveWithIdSerializer(serializers.Serializer):
    """Serializer for saving a workflow with a specific ID."""

    workflow = serializers.JSONField()
    workflow_id = serializers.UUIDField()

    def validate_workflow_id(self, value):
        """Validate that the workflow ID is a UUID."""
        if not isinstance(value, uuid.UUID):
            raise serializers.ValidationError("Workflow ID must be of type UUID")
        return value
